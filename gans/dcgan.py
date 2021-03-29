from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import paddle 
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.vision.datasets as dset
import paddle.vision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
paddle.seed(manualSeed)

dataroot = "/paddle/data/celeba/tmp/img_align_celeba"
output_path = 'output'
workers = 0
batch_size = 128
image_size = 64
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
ngf = 64
ndf = 64
epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

class Norm(transforms.BaseTransform):
    def __init__(self, keys=None):
        super(Norm, self).__init__(keys)
    def _apply_image(self, img):
        m = np.array(img).astype('float32') / 255.0
        return m.transpose((2, 0, 1))

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               Norm(),
                           ]))


dataloader = paddle.io.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=workers)
#paddle.set_device('gpu')

for batch_id, data in enumerate(dataloader):
    plt.figure(figsize=(15,15))
    try:
        for i in range(100):
            dt = data[0][i]
            image = dt.transpose((1,2,0)).numpy()
            plt.subplot(10, 10, i + 1)
            plt.imshow(image, vmin=-1, vmax=1)
            plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.suptitle('\n Training Images',fontsize=30)
        plt.show()
        plt.savefig('train_images.png')
        break
    except IOError:
        print(IOError)


@paddle.no_grad()
def normal_(x, mean=0., std=1.):
    temp_value = paddle.normal(mean, std, shape=x.shape)
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def uniform_(x, a=-1., b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x

@paddle.no_grad()
def constant_(x, value):
    temp_value = paddle.full(x.shape, value, x.dtype)
    x.set_value(temp_value)
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and classname.find('Conv') != -1:
        normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        normal_(m.weight, 1.0, 0.02)
        constant_(m.bias, 0)

# Generator Code
class Generator(nn.Layer):
    def __init__(self, ):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.Conv2DTranspose( nz, ngf * 8, 4, 1, 0, bias_attr=False),
            nn.BatchNorm2D(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2DTranspose(ngf * 8, ngf * 4, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2DTranspose( ngf * 4, ngf * 2, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Conv2DTranspose( ngf * 2, ngf, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2DTranspose( ngf, nc, 4, 2, 1, bias_attr=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.gen(x)


netG = Generator()
# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


class Discriminator(nn.Layer):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2D(nc, ndf, 4, 2, 1, bias_attr=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2D(ndf, ndf * 2, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(ndf * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2D(ndf * 2, ndf * 4, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(ndf * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2D(ndf * 4, ndf * 8, 4, 2, 1, bias_attr=False),
            nn.BatchNorm2D(ndf * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2D(ndf * 8, 1, 4, 1, 0, bias_attr=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)

netD = Discriminator()
netD.apply(weights_init)
print(netD)


# Initialize BCELoss function
loss = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = paddle.randn([64, nz, 1, 1], dtype='float32')

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(parameters=netD.parameters(), learning_rate=lr, beta1=beta1, beta2=0.999)
optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=lr, beta1=beta1, beta2=0.999)


losses = [[], []]
#plt.ion()
now = 0
for pass_id in range(epochs):
    for batch_id, (data,) in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        optimizerD.clear_grad()
        real_cpu = data
        bs_size = real_cpu.shape[0]
        label = paddle.full((bs_size, 1, 1, 1), real_label, dtype='float32')
        output = netD(real_cpu)
        errD_real = loss(output, label)
        errD_real.backward()

        noise = paddle.randn([bs_size, nz, 1, 1], 'float32')
        fake = netG(noise)
        label = paddle.full((bs_size, 1, 1, 1), fake_label, dtype='float32')
        output = netD(fake.detach())
        errD_fake = loss(output,label)
        errD_fake.backward()
        optimizerD.step()
        optimizerD.clear_grad()

        errD = errD_real + errD_fake
        losses[0].append(errD.numpy()[0])

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.clear_grad()
        noise = paddle.randn([bs_size, nz, 1, 1],'float32')
        fake = netG(noise)
        label = paddle.full((bs_size, 1, 1, 1), real_label, dtype=np.float32,)
        output = netD(fake)
        errG = loss(output,label)
        errG.backward()
        optimizerG.step()
        optimizerG.clear_grad()

        losses[1].append(errG.numpy()[0])
        if batch_id % 100 == 0:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            generated_image = netG(noise).numpy()
            imgs = []
            plt.figure(figsize=(15,15))
            try:
                for i in range(100):
                    image = generated_image[i].transpose()
                    image = np.where(image > 0, image, 0)
                    image = image.transpose((1,0,2))
                    plt.subplot(10, 10, i + 1)
                    plt.imshow(image, vmin=-1, vmax=1)
                    plt.axis('off')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplots_adjust(wspace=0.1, hspace=0.1)
                msg = 'Epoch ID={0} Batch ID={1} \n\n D-Loss={2} G-Loss={3}'.format(pass_id, batch_id, errD.numpy()[0], errG.numpy()[0])
                print(msg)
                plt.suptitle(msg,fontsize=20)
                plt.draw()
                plt.savefig('{}/{:04d}_{:04d}.png'.format(output_path, pass_id, batch_id), bbox_inches='tight')
                plt.pause(0.01)
            except IOError:
                print(IOError)
    paddle.save(netG.state_dict(), "work/generator.params")

plt.close()
