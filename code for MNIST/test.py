import os, time, pdb
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# G(z)
class generator1(nn.Module):
    # initializers
    def __init__(self, d=128, r=16):
        super(generator1, self).__init__()
        self.Econv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.Econv1_bn = nn.BatchNorm2d(d)
        self.Econv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.Econv2_bn = nn.BatchNorm2d(d*2)
        self.Econv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.Econv3_bn = nn.BatchNorm2d(d*4)
        self.Econv4 = nn.Conv2d(d * 4, d, 4, 1, 0)
        self.Econv4_bn = nn.BatchNorm2d(d)
        self.Econv5 = nn.Conv2d(d, r, 1, 1, 0)
        
        self.deconv0 = nn.ConvTranspose2d(r, d, 1, 1, 0)
        self.deconv0_bn = nn.BatchNorm2d(d)
        self.deconv1 = nn.ConvTranspose2d(d, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.Econv1_bn(self.Econv1(input)))
        x = F.relu(self.Econv2_bn(self.Econv2(x)))
        x = F.relu(self.Econv3_bn(self.Econv3(x)))
        x = F.relu(self.Econv4_bn(self.Econv4(x)))
        x = F.tanh(self.Econv5(x))
        v = x + (torch.round((x.data+1)/2)*2-1-x.data)
        
        x = F.relu(self.deconv0_bn(self.deconv0(v)))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.deconv4(x)

        return x, v

class generator2(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator2, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*4)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.deconv4(x)

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=16):
        super(discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(rate, d * 8, 1, 1, 0)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv4 = nn.Conv2d(d * 4, d*8, 4, 1, 0)
        self.conv5 = nn.Conv2d(d * 16, d * 4, 1, 1, 0)
        self.conv6 = nn.Conv2d(d * 4, 1, 1, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input1, input2):
        x = F.leaky_relu(self.conv1_1(input1), 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        y = F.leaky_relu(self.conv1_2(input2), 0.2)
        x = torch.cat([x, y], 1)

        x = F.leaky_relu(self.conv5(x), 0.2)
        x = self.conv6(x)
        #x = F.sigmoid(self.conv4(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def show_result(img, num_epoch = 0, show = False, save = False, path = 'result/'):

    G1.eval()
    G2.eval()
    img_mse, bitstream = G1(img)

    z_ = torch.randn((100, 100-rate)).view(-1, 100-rate, 1, 1)
    z_ = Variable(z_.cuda())
    z_ = torch.cat([bitstream.data, z_], 1)
    img_p = G2(z_)

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img_mse[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'P=+∞'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'mse.png')

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'input'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'input.png')

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(img_p[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'P=0'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'P=0.png')
    
    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((0.2*img_mse[k, 0]+0.8*img_p[k, 0]).cpu().data.numpy(), cmap='gray')

    label = '1-4'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'1-4.png')

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((0.4*img_mse[k, 0]+0.6*img_p[k, 0]).cpu().data.numpy(), cmap='gray')

    label = '2-3'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'2-3.png')

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((0.6*img_mse[k, 0]+0.8*img_p[k, 0]).cpu().data.numpy(), cmap='gray')

    label = '3-2'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'3-2.png')

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow((0.8*img_mse[k, 0]+0.2*img_p[k, 0]).cpu().data.numpy(), cmap='gray')

    label = '4-1'
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path+'4-1.png')
    



    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_mse(hist, show = False, save = False, path = 'Train_mse.png'):
    x = range(len(hist['G1_mse']))

    y1 = hist['G1_mse']
    y2 = hist['G2_mse']

    plt.plot(x, y1, label='G1_mse')
    plt.plot(x, y2, label='G2_mse')

    plt.xlabel('Epoch')
    plt.ylabel('MSE')

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 100
pretrained = True
rate = 4

# results save folder
root = 'results/'
model = 'MNIST_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
    
# data_loader
img_size = 32
transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor()
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=False)

# network
G1 = generator1(64, rate)
G2 = generator2(32)
#D = discriminator(16)
G1.load_state_dict(torch.load(root + model + 'generator1_param.pkl'))
G2.load_state_dict(torch.load(root + model + 'generator2_param.pkl'))
#D.load_state_dict(torch.load(root + model + 'discriminator_param.pkl'))
G1.cuda()
G2.cuda()
#D.cuda()

train_hist = {}
train_hist['G1_mse'] = []
train_hist['G2_mse'] = []

print('training start!')
start_time = time.time()
G1_mse = []
G2_mse = []
G14_mse = []
G23_mse = []
G32_mse = []
G41_mse = []

# learning rate decay

epoch_start_time = time.time()
for x_, y_ in train_loader:
    mini_batch = x_.size()[0]
        
    x_ = Variable(x_.cuda())
    x_mse_, bitstream = G1(x_)
    G1_loss = torch.mean((x_mse_ - x_)**2)
    G1_mse.append(G1_loss.data)

    z_ = torch.randn((mini_batch, 100 - rate)).view(-1, 100 - rate, 1, 1)
    z_ = Variable(z_.cuda())
    G2_input = torch.cat([bitstream.data, z_], 1)

    G_result = G2(G2_input)
    G_result = 1*G_result + 0*x_mse_
    g2_mse = torch.mean((G_result.data - x_)**2)
    G2_mse.append(g2_mse.data)
    
    G_x = 0.8*G_result + 0.2*x_mse_
    g2_mse = torch.mean((G_x.data - x_)**2)
    G14_mse.append(g2_mse.data)
    
    G_x = 0.6*G_result + 0.4*x_mse_
    g2_mse = torch.mean((G_x.data - x_)**2)
    G23_mse.append(g2_mse.data)
    
    G_x = 0.4*G_result + 0.6*x_mse_
    g2_mse = torch.mean((G_x.data - x_)**2)
    G32_mse.append(g2_mse.data)
    
    G_x = 0.2*G_result + 0.8*x_mse_
    g2_mse = torch.mean((G_x.data - x_)**2)
    G41_mse.append(g2_mse.data)

print('mse without P constraint: %.4f, mse with P=0: %.4f, %.4f, %.4f, %.4f, %.4f' % (torch.mean(torch.FloatTensor(G1_mse)), torch.mean(torch.FloatTensor(G2_mse)), torch.mean(torch.FloatTensor(G14_mse)), torch.mean(torch.FloatTensor(G23_mse)), torch.mean(torch.FloatTensor(G32_mse)), torch.mean(torch.FloatTensor(G41_mse))))
fixed_p = root + 'Fixed_results/' + model
show_result(x_[0:100,:,:,:], save=True, path=fixed_p)


