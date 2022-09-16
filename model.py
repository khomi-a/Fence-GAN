import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

import torch
import torch.nn as nn

from tqdm import tqdm
from shutil import rmtree

from losses import GenFGANLoss, DiscFGANLoss


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--alpha", type=float, default=0.5, help="losses: alpha")
parser.add_argument("--beta", type=float, default=15, help="losses: beta")
parser.add_argument("--gamma", type=float, default=0.1, help="losses: gamma")
parser.add_argument("--v_animate", type=int, default=1000, help="animation verbosity")
parser.add_argument("--fps", type=int, default=2, help="animation fps")
parser.add_argument("--make_video", type=bool, default=False, help="True if want to get video output")


opt = parser.parse_args()
print(opt)

Tensor = torch.FloatTensor


def real_data(n):
    return np.random.normal((20, 20), 3, [n, 2])


def noise_data(n):
    return np.random.normal(0, 8, [n, 2])


def animate(G, D, epoch, v_anim, path='./pictures/', hyperparams=None):
    plt.figure()
    xlist = np.linspace(0, 40, 40)
    ylist = np.linspace(0, 40, 40)
    X, Y = np.meshgrid(xlist, ylist)
    In = Tensor(np.array(np.meshgrid(xlist, ylist))).T.reshape(-1, 2)
    Out = D.forward(In)
    Z = Out.reshape(40, 40).detach().T
    c = ('#66B2FF', '#99CCFF', '#CCE5FF', '#FFCCCC', '#FF9999', '#FF6666')
    cp = plt.contourf(X, Y, Z, [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], colors=c)
    plt.colorbar(cp)

    fake_data_batch = G(Tensor(noise_data(250)))
    real = real_data(250)
    gx, gy = fake_data_batch[:, 0].detach(), fake_data_batch[:, 1].detach()
    rx, ry = real[:, 0], real[:, 1]
    # plotting the sample data, generated data
    plt.scatter(rx, ry, color='red')
    plt.scatter(gx, gy, color='blue')

    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.xticks([]), plt.yticks([])
    title = 'Epoch' + str(epoch)
    if hyperparams:
        (al, bt, gm) = hyperparams
        title += f'\n$\\alpha={al}, \\beta={bt}, \\gamma={gm}$'
    plt.title(title)
    plt.savefig(path + '/' + str(int(epoch / v_anim)).zfill(3) + '.png', dpi=500)
    plt.close()


def make_video(image_folder, fps=2001//opt.v_animate, video_name='my_video.mp4'):
    image_files = [os.path.join(image_folder, img)
                   for img in os.listdir(image_folder)
                   if img.endswith(".png")]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(video_name)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.activation = nn.ReLU()

    def forward(self, Input):
        x = self.activation(self.fc1(Input))
        x = self.activation(self.fc2(x))
        x = self.fc3(x) + Input
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return nn.Sigmoid()(x)


# Generator hyperparameters
alpha = opt.alpha
beta = opt.beta
# Discriminator hyperparameters
gamma = opt.gamma

# Loss function
gen_loss = GenFGANLoss(alpha_=alpha, beta_=beta)
disc_loss = DiscFGANLoss(gamma_=gamma)

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

# pretrain
batch_size = opt.batch_size


def pretrain():
    for epoch in range(20):
        real_data_batch = Tensor(real_data(batch_size))
        fake_data_batch = generator(Tensor(noise_data(batch_size)))

        optimizer_D.zero_grad()

        d_loss = disc_loss(discriminator(real_data_batch), discriminator(fake_data_batch))
        d_loss.backward()
        optimizer_D.step()


n_epochs = opt.n_epochs + 1
v_animate = opt.v_animate
pictures_path = './pictures' + f'_al={alpha}_bt={beta}_gm={gamma}'
if os.path.exists(pictures_path):
    rmtree(pictures_path)
os.makedirs(pictures_path)
hyperparameters = (alpha, beta, gamma)


def train():
    for i, epoch in enumerate(tqdm(range(n_epochs))):
        real_data_batch = Tensor(real_data(batch_size))
        fake_data_batch = generator(Tensor(noise_data(batch_size)))

        #  Train Discriminator

        optimizer_D.zero_grad()
        d_loss = disc_loss(discriminator(real_data_batch), discriminator(fake_data_batch))

        d_loss.backward()
        optimizer_D.step()

        #  Train Generator

        optimizer_G.zero_grad()
        fake_data_batch = generator(Tensor(noise_data(batch_size)))
        g_loss = gen_loss(discriminator(fake_data_batch), fake_data_batch)

        g_loss.backward()
        optimizer_G.step()

        if epoch % v_animate == 0:
            animate(generator, discriminator, epoch, v_animate, path=pictures_path, hyperparams=hyperparameters)


pretrain()
print('Pretraining done. Training started:')

try:
    train()
except RuntimeError:
    print('Didn\'t converge. Another try:')
    generator = Generator()
    discriminator = Discriminator()
    train()
print('Training finished')


if opt.make_video:
    make_video(image_folder=pictures_path, fps=opt.fps, video_name=f'video_al={alpha}_bt={beta}_gm={gamma}.mp4')
