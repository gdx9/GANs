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
            nn.BatchNorm2d(FEATURES_NUM * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(FEATURES_NUM*2, FEATURES_NUM*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(FEATURES_NUM * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(FEATURES_NUM*4, FEATURES_NUM*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(FEATURES_NUM * 8),
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
    generator_optimizer = optim.RMSprop(generator.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

    # train
    NUM_EPOCHS = 20
    WEIGHT_CLIP = 0.01
    CRITIC_ITERATIONS = 5

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
            fake = generator(noise).detach()

            critic_pred_real = critic(real)
            critic_pred_fake = critic(fake)
            critic_loss = -torch.mean(critic_pred_real) + torch.mean(critic_pred_fake)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # clip critic weights between -0.01, 0.01
            for param in critic.parameters():
                param.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

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
