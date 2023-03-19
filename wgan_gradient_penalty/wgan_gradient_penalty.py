#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 2023

@author: gdx9
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from tqdm import tqdm
import matplotlib.pyplot as plt

# set critic and generator classes
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        FEATURES_NUM = 16
        self.critic_module = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=FEATURES_NUM*1, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=FEATURES_NUM*1, out_channels=FEATURES_NUM * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(FEATURES_NUM * 2, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(FEATURES_NUM*2, FEATURES_NUM*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(FEATURES_NUM * 4, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(FEATURES_NUM*4, FEATURES_NUM*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(FEATURES_NUM * 8, affine=True),
            nn.LeakyReLU(0.2),

            nn.Conv2d(FEATURES_NUM * 8, 1, kernel_size=4, stride=2, padding=0)
        )

        for layer in self.critic_module:
            if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.critic_module(x)

class Generator(nn.Module):
    def __init__(self, channels_noise):
        super(Generator, self).__init__()

        FEATURES_NUM = 16
        self.gen_module = nn.Sequential(
            # 4x4
            nn.ConvTranspose2d(in_channels=channels_noise, out_channels=FEATURES_NUM*16,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(FEATURES_NUM*16),
            nn.ReLU(),

            # 8x8
            nn.ConvTranspose2d(FEATURES_NUM*16, FEATURES_NUM*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(FEATURES_NUM*8),
            nn.ReLU(),

            # 16x16
            nn.ConvTranspose2d(FEATURES_NUM*8, FEATURES_NUM*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(FEATURES_NUM*4),
            nn.ReLU(),

            # 32x32
            nn.ConvTranspose2d(FEATURES_NUM*4, FEATURES_NUM*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(FEATURES_NUM*2),
            nn.ReLU(),

            nn.ConvTranspose2d(FEATURES_NUM * 2,
                               1,# 1-channel
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        for layer in self.gen_module:
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)

    def forward(self, x):
        return self.gen_module(x)

def clac_gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = (alpha * real + (1 - alpha) * fake)#.requires_grad_(True)

    prob_interpolated = critic(interpolated_images)

    # gradient of probs with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=prob_interpolated,
        grad_outputs=torch.ones_like(prob_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_penalty = torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
    return gradient_penalty

if __name__ == '__main__':
    # prepare data
    IMAGE_SIZE = 64
    transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    BATCH_SIZE = 32
    dataset = datasets.MNIST(root="./data", transform=transform, download=False)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    # prepare generator and critic
    device = "cuda" if torch.cuda.is_available() else "cpu"

    NOISE_DIM = 100
    generator = Generator(NOISE_DIM).to(device)
    critic = Critic().to(device)

    LEARNING_RATE = 1e-4
    generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))# (0.0, 0.9)
    critic_optimizer    = optim.Adam(critic.parameters(),    lr=LEARNING_RATE, betas=(0.5, 0.999))

    # train
    NUM_EPOCHS = 5
    CRITIC_ITERATIONS = 5
    LAMBDA_GP = 10

    generator.train()
    critic.train()

    for epoch in range(1,NUM_EPOCHS+1):
        epoch_critic_loss = 0.
        epoch_generator_loss = 0.

        iter_num = 0
        for real, _ in tqdm(data_loader):
            real = real.to(device)
            current_batch_size = len(real)
            iter_num += 1

            # descriminator
            noise = torch.randn(current_batch_size, NOISE_DIM, 1, 1, device=device)
            fake = generator(noise)#.detach()

            critic_pred_real = critic(real)
            critic_pred_fake = critic(fake)
            gradient_penalty = clac_gradient_penalty(critic, real, fake, device=device)

            critic_loss = -torch.mean(critic_pred_real) + torch.mean(critic_pred_fake) + LAMBDA_GP * gradient_penalty

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            epoch_critic_loss += critic_loss.item()

            if iter_num % CRITIC_ITERATIONS == 0:
                # generator
                fake = generator(noise)
                critic_pred_fake = critic(fake)

                generator_loss = -torch.mean(critic_pred_fake)
                generator_optimizer.zero_grad()
                generator_loss.backward()
                generator_optimizer.step()

                epoch_generator_loss += generator_loss.item()

        # show epoch results
        epoch_generator_loss /= (len(data_loader) / CRITIC_ITERATIONS)
        epoch_critic_loss /= len(data_loader)
        print(f"Epoch {epoch}: Generator loss: {epoch_generator_loss}, critic loss: {epoch_critic_loss}")

        fake = (fake + 1) / 2
        image_grid = make_grid(fake.detach().cpu()[:10], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
        
