#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 2023

@author: gdx9
"""

import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# model classes
class Generator(nn.Module):
    def __init__(self, noize_dim):
        super(Generator, self).__init__()
        
        self.gen = nn.Sequential(
            # 1
            nn.ConvTranspose2d(noize_dim, 256, kernel_size=3, stride=2, padding=0),# 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=0),# 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 3
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),# 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 4
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=0),# 32x32
            nn.Tanh()
        )
        for layer in self.gen:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.normal_(layer.weight, 0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.normal_(layer.weight, 0.0, 0.02)
                torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, noise):
        return self.gen(noise)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.dis = nn.Sequential(
            # 1
            nn.Conv2d(1, 16, kernel_size=4, stride=2),# 1-channel
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3
            nn.Conv2d(32, 1, kernel_size=4, stride=2)
        )
        
        for layer in self.dis:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                torch.nn.init.normal_(layer.weight, 0.0, 0.02)
            if isinstance(layer, nn.BatchNorm2d):
                torch.nn.init.normal_(layer.weight, 0.0, 0.02)
                torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, image):
        res = self.dis(image)
        return res.view(len(res), -1)

if __name__ == '__main__':
    # prepare data
    BATCH_SIZE = 128

    data_loader = DataLoader(
        MNIST('./data', download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),# [-1 ; 1] for Tanh actication
        ])),
        batch_size=BATCH_SIZE,
        shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("cuda available:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')

    # prepare model
    lr = 2e-4
    NOISE_DIM = 100
    gen = Generator(NOISE_DIM).to(device)
    dis = Discriminator().to(device)

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    dis_opt = torch.optim.Adam(dis.parameters(), lr=lr, betas=(0.5, 0.999))

    # train model
    EPOCHS = 10
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1,EPOCHS+1):
        epoch_discriminator_loss = 0.
        epoch_generator_loss = 0.

        for real, _ in tqdm(data_loader):
            real = real.to(device)
            current_batch_size = len(real)
            
            # descriminator
            
            fake_noise = torch.randn(current_batch_size, NOISE_DIM, 1, 1, device=device)
            fake = gen(fake_noise)
            fake_pred_dis = dis(fake.detach())
            fake_loss_dis = criterion(fake_pred_dis, torch.zeros_like(fake_pred_dis))
            
            real_pred_dis = dis(real)
            real_loss_dis = criterion(real_pred_dis, torch.ones_like(real_pred_dis))
            dis_loss = (fake_loss_dis + real_loss_dis) / 2
            
            dis_opt.zero_grad()
            dis_loss.backward(retain_graph=True)
            dis_opt.step()
            
            epoch_discriminator_loss += dis_loss.item()
            
            # generator
            #fake = gen(fake_noise)
            fake_pred_dis = dis(fake)
            gen_loss = criterion(fake_pred_dis, torch.ones_like(fake_pred_dis))
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()
            
            epoch_generator_loss += gen_loss.item()
            
        # show epoch results
        epoch_generator_loss /= len(data_loader)
        epoch_discriminator_loss /= len(data_loader)
        print(f"Epoch {epoch}: Generator loss: {epoch_generator_loss}, discriminator loss: {epoch_discriminator_loss}")
        
        fake = (fake + 1) / 2
        image_grid = make_grid(fake.detach().cpu()[:10], nrow=5)
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
            
    # save generator
    model_name = "generator_dcgan.onnx"
    torch.onnx.export(gen.to('cpu'),
                    torch.randn(1, NOISE_DIM, 1, 1),# input example with size
                    model_name,
                    verbose=False,
                    input_names=["actual_input"],
                    output_names=["output"],
                    export_params=True)
    
    print("done")
