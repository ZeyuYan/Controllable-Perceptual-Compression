import os, time, pdb
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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

def show_result(num_epoch, show = False, save = False, path = 'result.png'):

    G.eval()
    test_images = G(fixed_z_)
    G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    #y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    #plt.plot(x, y2, label='G_loss')

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
    y3 = hist['2G1_mse']

    plt.plot(x, y1, label='G1_mse')
    plt.plot(x, y2, label='G2_mse')
    plt.plot(x, y3, label='2*G1_mse')

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
batch_size = 128
lr = 0.001
train_epoch = 100
lambda_gp = 10
beta = 0.99
pretrained = False
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
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G1 = generator1(64, rate)
G2 = generator2(32)
D = discriminator(16)
if pretrained == True:
    G1.load_state_dict(torch.load(root + model + 'generator1_param.pkl'))
    G2.load_state_dict(torch.load(root + model + 'generator2_param.pkl'))
    D.load_state_dict(torch.load(root + model + 'discriminator_param.pkl'))
else:
    G1.weight_init(mean=0.0, std=0.02)
    G2.weight_init(mean=0.0, std=0.02)
    D.weight_init(mean=0.0, std=0.02)
G1.cuda()
G2.cuda()
D.cuda()

# Adam optimizer
#G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
#D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

G1_optimizer = optim.RMSprop(G1.parameters(), lr=lr/10)
G2_optimizer = optim.RMSprop(G2.parameters(), lr=lr/10)
D_optimizer = optim.RMSprop(D.parameters(), lr=lr)

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['G1_mse'] = []
train_hist['G2_mse'] = []
train_hist['2G1_mse'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('training start!')
start_time = time.time()
for epoch in range(train_epoch):
    D_losses = []
    G_losses = []
    G1_mse = []
    G1_2_mse = []
    G2_mse = []

    # learning rate decay
    if (epoch+1) == 100:
        G1_optimizer.param_groups[0]['lr'] /= 10
        G2_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")
    '''
    if (epoch+1) == 40:
        G1_optimizer.param_groups[0]['lr'] /= 2
        G2_optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 4
        print("learning rate change!")
    '''

    epoch_start_time = time.time()
    for x_, y_ in train_loader:
        # train generator1 G1
        G1.zero_grad()
        mini_batch = x_.size()[0]
        
        x_ = Variable(x_.cuda())
        x_mse_, bitstream = G1(x_)
        G1_loss = torch.mean((x_mse_ - x_)**2)

        G1_loss.backward()
        G1_optimizer.step()

        G1_mse.append(G1_loss.data)
        G1_2_mse.append(2*G1_loss.data)

        if (epoch+1) > 0:
            # train discriminator D
            D.zero_grad()

            D_result = D(x_, bitstream.data).squeeze()
            D_real_loss = -D_result.mean()

            z_ = torch.randn((mini_batch, 100 - rate)).view(-1, 100 - rate, 1, 1)
            z_ = Variable(z_.cuda())
            G2_input = torch.cat([bitstream.data, z_], 1)

            G_result = G2(G2_input)
            D_result = D(G_result.data, bitstream.data).squeeze()

            D_fake_loss = D_result.mean()
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            #gradient penalty
            D.zero_grad()
            alpha = torch.rand(x_.size(0), 1, 1, 1)
            alpha1 = alpha.cuda().expand_as(x_)
            interpolated1 = Variable(alpha1 * x_.data + (1 - alpha1) * G_result.data, requires_grad=True)
            interpolated2 = Variable(bitstream.data, requires_grad=True)

            #alpha2 = alpha.cuda().expand_as(bitstream1)
            #interpolated2 = Variable(alpha2 * bitstream1.data + (1 - alpha2) * bitstream2.data, requires_grad=True)
            out = D(interpolated1, interpolated2).squeeze()

            grad = torch.autograd.grad(outputs=out,
                                       inputs=[interpolated1, interpolated2],
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            gp_loss = lambda_gp * d_loss_gp

            gp_loss.backward()
            D_optimizer.step()

            # train generator G
            D.zero_grad()
            G2.zero_grad()

            z_ = torch.randn((mini_batch, 100 - rate)).view(-1, 100 - rate, 1, 1)
            z_ = Variable(z_.cuda())
            G2_input = torch.cat([bitstream.data, z_], 1)

            G_result = G2(G2_input)
            D_result = D(G_result, bitstream.data).squeeze()
            
            #mse_loss = torch.sqrt(torch.sum((G_result - x_mse_.data)**2, dim=[1,2,3]))
            mse_loss = torch.mean((G_result - x_mse_.data)**2, dim=[1,2,3])

            G_train_loss = - beta * D_result.mean() + (1-beta) * mse_loss.mean()

            G_train_loss.backward()
            G2_optimizer.step()

            G_losses.append(G_train_loss.data)
            g2_mse = torch.mean((G_result.data - x_)**2)
            G2_mse.append(g2_mse.data)
        else:
            arr = [i for i in range(0, mini_batch)]
            np.random.shuffle(arr)
            bit_f = bitstream[arr,:,:,:].data
            D_result = D(x_, bitstream.data).squeeze()
            D_real_loss = -D_result.mean()

            D_result = D(x_, bit_f.data).squeeze()
            D_fake_loss = D_result.mean()
            D_fake_score = D_result.data.mean()

            D_train_loss = D_real_loss + D_fake_loss

            D_train_loss.backward()
            D_optimizer.step()

            D_losses.append(D_train_loss.data)

            #gradient penalty
            D.zero_grad()
            alpha = torch.rand(x_.size(0), 1, 1, 1)
            alpha1 = alpha.cuda().expand_as(x_)
            interpolated1 = Variable(x_.data, requires_grad=True)
            #interpolated2 = Variable(bit2.data, requires_grad=True)

            alpha2 = alpha.cuda().expand_as(bitstream)
            interpolated2 = Variable(alpha2 * bitstream.data + (1 - alpha2) * bit_f.data, requires_grad=True)
            out = D(interpolated1, interpolated2).squeeze()

            grad = torch.autograd.grad(outputs=out,
                                       inputs=[interpolated1, interpolated2],
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            gp_loss = lambda_gp * d_loss_gp

            gp_loss.backward()
            D_optimizer.step()

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, G1_mse: %.4f, G2_mse: %.4f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G1_mse)), torch.mean(torch.FloatTensor(G2_mse))))
    
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    #show_result((epoch+1), save=True, path=fixed_p)
    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
    train_hist['G1_mse'].append(torch.mean(torch.FloatTensor(G1_mse)))
    train_hist['2G1_mse'].append(torch.mean(torch.FloatTensor(G1_2_mse)))
    train_hist['G2_mse'].append(torch.mean(torch.FloatTensor(G2_mse)))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), train_epoch, total_ptime))
print("Training finish!... save training results")
torch.save(G1.state_dict(), root + model + 'generator1_param.pkl')
torch.save(G2.state_dict(), root + model + 'generator2_param.pkl')
torch.save(D.state_dict(), root + model + 'discriminator_param.pkl')
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')
show_train_mse(train_hist, save=True, path=root + model + 'train_mse.png')
#ave(root + model + 'generation_animation.gif', images, fps=5)
np.savetxt(root+model+'G1_mse.txt', train_hist['G1_mse'], delimiter=" ")
np.savetxt(root+model+'G2_mse.txt', train_hist['G2_mse'], delimiter=" ")

