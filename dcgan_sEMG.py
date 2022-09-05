from dis import dis
from turtle import forward
from sklearn import datasets
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets as datasets
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from DataRetrieval import dataset_mat_CSL
import os
import torchvision
import numpy as np
from Dataset_pytorch import Dataset_pytorch
class Discriminator(nn.Module):
    def __init__(self,channels_img,features_d,image_size1,image_size2,num_classes) -> None:
        super(Discriminator,self).__init__()
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.num_classes = num_classes
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img+1,features_d,kernel_size=3, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,3,2,1),
            self._block(features_d*2,features_d*4,3,2,1),
            self._block(features_d*4,features_d*8,3,2,1),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
        self.embbed = nn.Embedding(self.num_classes,self.image_size1*self.image_size2)
    
    def _block(self,in_channels,out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x,labels):
        emb = self.embbed(labels).view(labels.size(0),1,self.image_size1,self.image_size2)
        x = torch.cat((x,emb),1)
        x = self.disc(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class Generator(nn.Module):
    def __init__(self,z_dim, channels_img, features_g, image_size1,image_size2,emb_size,num_classes) -> None:
        super(Generator,self).__init__()
        self.image_size1 = image_size1
        self.image_size2 = image_size2
        self.emb_size = emb_size
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.embbed = nn.Embedding(self.num_classes,self.emb_size)
        self.l1 = nn.Sequential(nn.Linear(z_dim+self.emb_size, z_dim * image_size1 * image_size2))
        self.gen = nn.Sequential(
            self._block(z_dim,features_g*16,3,1,1),
            self._block(features_g*16,features_g*8, 3,1,1),
            self._block(features_g*8,features_g*4, 3,1,1),
            self._block(features_g*4,features_g*2, 3,1,1),
            nn.ConvTranspose2d(
                features_g*2,channels_img,kernel_size=3,stride=1,padding=1
            ),
            nn.Tanh()
        )

    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self,x,labels):
        labels = self.embbed(labels)
        x = torch.cat((x,labels),1)
        x = self.l1(x)
        x = x.view(x.shape[0], self.z_dim, self.image_size1, self.image_size2)
        x = self.gen(x)
        return x
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

def test():
    N,in_channels,H,W = 8,1,8,32
    z_dim = 100
    x = torch.randn((N,in_channels,H,W))
    disc = Discriminator(in_channels,64)
    initialize_weights(disc)
    assert disc(x).shape == (N,1)
    gen = Generator(z_dim,in_channels,64,H,W)
    initialize_weights(gen)
    z = torch.randn((N,z_dim))
    assert gen(z).shape == (N, in_channels,H,W)
    print("Success")

# test()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE1 = 8
IMAGE_SIZE2 = 24
CHANNELS_IMG = 1
Z_DIM = 100
EMBBED_SIZE = 100
NUM_EPOCHS = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
NUM_CLASSES = 8

data = None

label = None
# prepare dataset
def key(element):
    a = element.split(".")
    return int(a[0][4:])
for gest in range(1,9):
    for i in range(1,6):
        if data is None:
            data = dataset_mat_CSL(f"data/clean/001-00{gest}-00{i}.mat",ICE=True)
            label = np.repeat(gest-1,data.shape[0])
        else:
            temp = dataset_mat_CSL(f"data/clean/001-00{gest}-00{i}.mat",ICE=True)
            if temp.shape[1] == 193:
                temp = temp[:,:-1]
            data = np.concatenate((data,temp),axis=0)
            label = np.concatenate((label,np.repeat(gest-1,temp.shape[0])),axis=0)
X_train = data
y_train = label
mean = np.mean(X_train)
std = np.std(X_train)
dataset = Dataset_pytorch(X_train,y_train,mean=mean,std=std)


loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=16)


gen = Generator(Z_DIM,CHANNELS_IMG,FEATURES_GEN,IMAGE_SIZE1,IMAGE_SIZE2,EMBBED_SIZE,NUM_CLASSES).to(device)

disc = Discriminator(CHANNELS_IMG,FEATURES_DISC,IMAGE_SIZE1,IMAGE_SIZE2,NUM_CLASSES).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(),lr=LEARNING_RATE,betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(128,Z_DIM).to(device)
gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real,labels) in enumerate(loader):
        real = real.to(device)
        labels = labels.to(device)
        noise = torch.randn((real.size(0),Z_DIM)).to(device)
        fake = gen(noise,labels)
        disc_real  = disc(real,labels).reshape(-1)
        loss_disc_real = criterion(disc_real,torch.ones_like(disc_real))
        disc_fake  = disc(fake,labels).reshape(-1)
        loss_disc_fake = criterion(disc_fake,torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        output = disc(fake,labels).reshape(-1)
        loss_gen = criterion(output,torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
            )


            torch.save(gen.state_dict(),f"weight/{loss_gen:.4f}")
labels = torch.randint(0,9,(128,))
labels = labels.to(device)
fake = gen(fixed_noise,labels)
print(fake.shape)
print(labels.shape)
torch.save(fake,"sample")
torch.save(labels,"labels")