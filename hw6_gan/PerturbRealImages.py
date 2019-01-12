import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
from torch.autograd import Variable


class MyDiscriminator(nn.Module):

    def __init__(self):
        super(MyDiscriminator, self).__init__()
        num_feature = 128
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=2)
        self.conv3 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=1)
        self.conv8 = nn.Conv2d(in_channels=num_feature, out_channels=num_feature,
                               kernel_size=3, padding=1, stride=2)

        self.ln1 = nn.LayerNorm([num_feature, 32, 32])
        self.ln2 = nn.LayerNorm([num_feature, 16, 16])
        self.ln3 = nn.LayerNorm([num_feature, 16, 16])
        self.ln4 = nn.LayerNorm([num_feature, 8, 8])
        self.ln5 = nn.LayerNorm([num_feature, 8, 8])
        self.ln6 = nn.LayerNorm([num_feature, 8, 8])
        self.ln7 = nn.LayerNorm([num_feature, 8, 8])
        self.ln8 = nn.LayerNorm([num_feature, 4, 4])

        self.fc1 = nn.Linear(in_features=num_feature, out_features=1)
        self.fc10 = nn.Linear(in_features=num_feature, out_features=10)

    def forward(self, x):
        out = F.leaky_relu(self.ln1(self.conv1(x)))
        out = F.leaky_relu(self.ln2(self.conv2(out)))
        out = F.leaky_relu(self.ln3(self.conv3(out)))
        out = F.leaky_relu(self.ln4(self.conv4(out)))
        out = F.leaky_relu(self.ln5(self.conv5(out)))
        out = F.leaky_relu(self.ln6(self.conv6(out)))
        out = F.leaky_relu(self.ln7(self.conv7(out)))
        out = F.leaky_relu(self.ln8(self.conv8(out)))
        out = F.max_pool2d(out, kernel_size=4, padding=0, stride=4)
        out = out.view(-1, 128*1*1)
        out1 = self.fc1(out)
        out10 = self.fc10(out)
        outlist = [out1, out10]
        return outlist


def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


batch_size = 128
transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = enumerate(testloader)

model = torch.load('./cifar10.model')
model.cuda()
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
torch.backends.cudnn.benchmark = True
model.eval()

batch_idx, (X_batch, Y_batch) = testloader.__next__()
X_batch = Variable(X_batch, requires_grad=True).cuda()
Y_batch_alternate = (Y_batch + 1) % 10
Y_batch_alternate = Variable(Y_batch_alternate).cuda()
Y_batch = Variable(Y_batch).cuda()


# save real images
samples = X_batch.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0, 2, 3, 1)

fig = plot(samples[0:100])
plt.savefig('visualization/real_images.png', bbox_inches='tight')
plt.close(fig)

_, output = model(X_batch)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)


# slightly jitter all input images
criterion = nn.CrossEntropyLoss(reduce=False)
loss = criterion(output, Y_batch_alternate)

gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                          grad_outputs=torch.ones(loss.size()).cuda(),
                          create_graph=True, retain_graph=False, only_inputs=True)[0]

# save gradient jitter
gradient_image = gradients.data.cpu().numpy()
gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
gradient_image = gradient_image.transpose(0,2,3,1)
fig = plot(gradient_image[0:100])
plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
plt.close(fig)


# jitter input image
gradients[gradients>0.0] = 1.0
gradients[gradients<0.0] = -1.0

gain = 8.0
X_batch_modified = X_batch - gain*0.007843137*gradients
X_batch_modified[X_batch_modified>1.0] = 1.0
X_batch_modified[X_batch_modified<-1.0] = -1.0

# evaluate new fake images
_, output = model(X_batch_modified)
prediction = output.data.max(1)[1] # first column has actual prob.
accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
print(accuracy)

# save fake images
samples = X_batch_modified.data.cpu().numpy()
samples += 1.0
samples /= 2.0
samples = samples.transpose(0,2,3,1)

fig = plot(samples[0:100])
plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
plt.close(fig)
print("finish")
