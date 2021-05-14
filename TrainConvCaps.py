import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, interpolate
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import glob
import math
from functools import partial
from torchvision import transforms, utils
import random
import os
from datetime import datetime

class EvapoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, split=False, train_or_test = True, train=0.8, climate=None):
        
        self.file_names_orig = glob.glob("Joined_Data_16x16_Cloudy/*.npy")
        self.vegs = ['BSV', 'CRO', 'CSH', 'DBF', 'ENF', 'GRA', 'MF', 'OSH', 'SAV', 'WAT', 'WET', 'WSA']
        self.clim = ['', 'Bsh', 'Bsk', 'Bwh', 'Bwk', 'Cfa', 'Csa', 'Csb', 'Cwa', 'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dwb', 'Dwc', 'ET']
        
        
        self.file_names = []
#         for i in self.file_names_orig:
#             if not (('US-A03' in i) or ('US-A10' in i) or ('US-KS4' in i) or ('US-Myb' in i) or ('US-NGB' in i) or ('US-Sne' in i) or 
#             ('US-StJ' in i) or ('US-Tw1' in i) or ('US-Tw3' in i) or ('US-Tw4' in i) or ('US-UMB' in i) or ('US-UMd' in i) or 
#             ('US-xBA' in i) or ('US-xDJ' in i) or ('US-xSE' in i)):
#                 self.file_names.append(i)
                
        for i in self.file_names_orig:
            if (climate is not None):
                if (climate in i.split("_")[-6]):
                    self.file_names.append(i)
            else:
                self.file_names.append(i)
        
        print("Dataset Length " + str(len(self.file_names)))
        
        
        if split:
            random.Random(18).shuffle(self.file_names)
            self.length = len(self.file_names)
            indx = round(self.length * train)
            if train_or_test:
                self.file_names = self.file_names[:indx]
            else:
                self.file_names = self.file_names[indx:]
                
        self.transform = transforms.Compose([
            transforms.Normalize(mean=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])]) 
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img = self.transform(torch.from_numpy(np.load(self.file_names[idx]).astype(float)[0])[:11])
        
        img = interpolate(img.reshape((1, img.size(0), img.size(1), img.size(2))) , size=32)
        
        date_time_obj = datetime.strptime(self.file_names[idx].split("_")[-4] + '_' + self.file_names[idx].split("_")[-3] + '_' + self.file_names[idx].split("_")[-2], '%Y_%m_%d')
        day_of_year = date_time_obj.timetuple().tm_yday
        day_sin = torch.tensor([np.sin(2 * np.pi * day_of_year/364.0)])
        day_cos = torch.tensor([np.cos(2 * np.pi * day_of_year/364.0)])
        
        img = img.reshape(img.size(1), img.size(2), img.size(3))
#         et = max(float(self.file_names[idx].split('_')[-1].replace('.npy', '')), float(0))
        et = float(self.file_names[idx].split('_')[-1].replace('.npy', ''))
        veg = torch.nn.functional.one_hot(torch.tensor(self.vegs.index(self.file_names[idx].split("_")[-7])), num_classes=12)
        
        lat = float(self.file_names[idx].split('_')[-10])
        lon = float(self.file_names[idx].split('_')[-9])
        x_coord = torch.tensor([np.cos(lat) * np.cos(lon)])
        y_coord = torch.tensor([np.cos(lat) * np.sin(lon)])
        z_coord = torch.tensor([np.sin(lat)])
       
        elev = torch.tensor([float(self.file_names[idx].split('_')[-8])/20310.0]) # Tallest Point in US is 20310
        
        return img, et, veg, day_sin, day_cos, x_coord, y_coord, z_coord, elev
    
class convolutionalCapsule(nn.Module):
    def __init__(self, in_capsules, out_capsules, in_channels, out_channels, stride=1, padding=2,
                 kernel=5, num_routes=3, nonlinearity='sqaush', batch_norm=False, dynamic_routing='local', cuda=False):
        super(convolutionalCapsule, self).__init__()
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.in_capsules = in_capsules
        self.out_capsules = out_capsules
        self.out_channels = out_channels
        self.nonlinearity = nonlinearity
        self.batch_norm = batch_norm
        self.bn = nn.BatchNorm2d(in_capsules*out_capsules*out_channels)
        self.conv2d = nn.Conv2d(kernel_size=(kernel, kernel), stride=stride, padding=padding,
                                in_channels=in_channels, out_channels=out_channels*out_capsules)
        self.dynamic_routing = dynamic_routing
        self.cuda = cuda

    def forward(self, x):
        batch_size = x.size(0)
        in_width, in_height = x.size(3), x.size(4)
        x = x.view(batch_size*self.in_capsules, self.in_channels, in_width, in_height)
        u_hat = self.conv2d(x)

        out_width, out_height = u_hat.size(2), u_hat.size(3)

        # batch norm layer
        if self.batch_norm:
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = u_hat.view(batch_size, self.in_capsules * self.out_capsules * self.out_channels, out_width, out_height)
            u_hat = self.bn(u_hat)
            u_hat = u_hat.view(batch_size, self.in_capsules, self.out_capsules*self.out_channels, out_width, out_height)
            u_hat = u_hat.permute(0,1,3,4,2).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)

        else:
            u_hat = u_hat.permute(0,2,3,1).contiguous()
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules*self.out_channels)
            u_hat = u_hat.view(batch_size, self.in_capsules, out_width, out_height, self.out_capsules, self.out_channels)


        b_ij = Variable(torch.zeros(1, self.in_capsules, out_width, out_height, self.out_capsules))
        if self.cuda:
            b_ij = b_ij.cuda()
        for iteration in range(self.num_routes):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(5)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)


            if (self.nonlinearity == 'relu') and (iteration == self.num_routes - 1):
                v_j = F.relu(s_j)
            elif (self.nonlinearity == 'leakyRelu') and (iteration == self.num_routes - 1):
                v_j = F.leaky_relu(s_j)
            else:
                v_j = self.squash(s_j)

            v_j = v_j.squeeze(1)

            if iteration < self.num_routes - 1:
                temp = u_hat.permute(0, 2, 3, 4, 1, 5)
                temp2 = v_j.unsqueeze(5)
                a_ij = torch.matmul(temp, temp2).squeeze(5) # dot product here
                a_ij = a_ij.permute(0, 4, 1, 2, 3)
                b_ij = b_ij + a_ij.mean(dim=0)

        v_j = v_j.permute(0, 3, 4, 1, 2).contiguous()

        return v_j

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        output_tensor = squared_norm * input_tensor / ((1. + squared_norm) * torch.sqrt(squared_norm))
        return output_tensor
    
    
    
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_input = nn.Sequential(
            nn.Conv2d(11, 256, 3, 1, padding=1, bias=False)
        )
        
        self.conv_caps = nn.Sequential(
            convolutionalCapsule(in_capsules=16, out_capsules=8, in_channels=16, out_channels=32,
                                  stride=2, padding=1, kernel=3,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True),
            convolutionalCapsule(in_capsules=8, out_capsules=4, in_channels=32, out_channels=64,
                                  stride=2, padding=2, kernel=5,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True),
            convolutionalCapsule(in_capsules=4, out_capsules=2, in_channels=64, out_channels=128,
                                  stride=2, padding=3, kernel=7,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True),
            convolutionalCapsule(in_capsules=2, out_capsules=1, in_channels=128, out_channels=256,
                                  stride=2, padding=4, kernel=9,
                                  nonlinearity='sqaush', batch_norm=True,
                                  dynamic_routing='local', cuda=True),
            
        )
        
#         self.conv_caps = nn.Sequential(
#             convolutionalCapsule(in_capsules=16, out_capsules=8, in_channels=16, out_channels=32,
#                                   stride=2, padding=1, kernel=3,
#                                   nonlinearity='sqaush', batch_norm=True,
#                                   dynamic_routing='local', cuda=True),
#             convolutionalCapsule(in_capsules=8, out_capsules=4, in_channels=32, out_channels=64,
#                                   stride=1, padding=2, kernel=5,
#                                   nonlinearity='sqaush', batch_norm=True,
#                                   dynamic_routing='local', cuda=True),
#             convolutionalCapsule(in_capsules=4, out_capsules=2, in_channels=64, out_channels=128,
#                                   stride=1, padding=2, kernel=5,
#                                   nonlinearity='sqaush', batch_norm=True,
#                                   dynamic_routing='local', cuda=True),
#             convolutionalCapsule(in_capsules=2, out_capsules=1, in_channels=128, out_channels=256,
#                                   stride=1, padding=2, kernel=5,
#                                   nonlinearity='sqaush', batch_norm=True,
#                                   dynamic_routing='local', cuda=True),
            
#         )
        
        
        self.metadata_network = torch.nn.Sequential(
            torch.nn.Linear(18, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64)
        )
        
        self.linear_output = nn.Linear(1024 + 64, 1)
        
        
       
    def forward(self, x, metadata):
        x = self.conv_input(x)
        x = x.view(x.size(0), 16, 16, x.size(2), x.size(3))
#         x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        x = self.conv_caps(x)
        x = x.view(x.size(0), -1)
#         print("HERE")
#         print(x.shape)
        
        y = self.metadata_network(metadata)
        x = self.linear_output(torch.cat((x, y), dim=1))
        
        return x

    
class ConvCapsNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.Encoder = Encoder()
        
#     seq_len, batch, input_size
    def forward(self, x, y):
        out = self.Encoder(x, y)
        return out.flatten()

    
    
class TrainConvCapsNet():

    def __init__(self, epochs=300, batch_size=16, torch_type=torch.float32, random_seed=18, climate=None):
        super(TrainConvCapsNet, self).__init__()
        
        torch.cuda.manual_seed(18)
        torch.cuda.manual_seed_all(18)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        self.torch_type = torch_type
        
        self.mse = torch.nn.MSELoss()
        self.model = ConvCapsNet().to(self.device, dtype=torch.float32)
        
        self.dataset = EvapoDataset(climate=climate)
        
        self.dataset_size = len(self.dataset)
        self.indices = list(range(self.dataset_size))
        self.validation_split = 0.2
        self.shuffle_dataset = True
        self.random_seed = random_seed
        self.split = int(np.floor(self.validation_split * self.dataset_size))
        if self.shuffle_dataset :
            np.random.seed(self.random_seed)
            np.random.shuffle(self.indices)
        self.train_indices, self.test_indices = self.indices[self.split:], self.indices[:self.split]

        # Creating PT data samplers and loaders:
        self.train_sampler = torch.utils.data.SubsetRandomSampler(self.train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(self.test_indices)

        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                        sampler=self.train_sampler, drop_last=True, num_workers=5)
        self.test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                       sampler=self.test_sampler, num_workers=5)
        
        #self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        #self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.1)
#         self.opt = torch.optim.Adagrad(self.model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
#         self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25, gamma=0.5)
        self.opt = torch.optim.Adagrad(self.model.parameters(), lr_decay=0.01)
    
    def train(self):
        for epoch in range(self.epochs):
            for ind, (img_seq, et, veg, day_sin, day_cos, x, y, z, elev) in enumerate(self.train_loader):
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                et = et.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x = x.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.float32)
                z = z.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                output = self.model(img_seq, torch.cat((veg, day_sin, day_cos, x, y, z, elev), dim=1))
                loss = self.mse(output, et)
                loss.backward()
                self.opt.step()
#             self.sched.step()
            self.accuracy = self.test(epoch)
        torch.save(self.model, "checkpoints/ConvCapsNet_" + climate + "_" + str(self.random_seed) + ".pt" )
        return self.accuracy
        
        
    def test(self, epoch):
        with torch.no_grad():
            correct = 0
            counter = 0
            for img_seq, et, veg, day_sin, day_cos, x, y, z, elev in self.test_loader:
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                et = et.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x = x.to(device=self.device, dtype=torch.float32)
                y = y.to(device=self.device, dtype=torch.float32)
                z = z.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                output = self.model(img_seq, torch.cat((veg, day_sin, day_cos, x, y, z, elev), dim=1))
                correct += (torch.sum(torch.abs((output-et))))
                counter += output.shape[0]
            print("Epoch " + str(epoch) + " Accuracy: " + str(round(float(correct.sum() / counter), 4)))
            return str(round(float(correct.sum() / counter), 4))
            
                

if __name__ == '__main__':
    from argparse import ArgumentParser

    #parser = ArgumentParser()
    # add PROGRAM level args
    #.add_argument('--geohash', type=str, default=None)
    #args = parser.parse_args()    
    os.makedirs("checkpoints", exist_ok = True)
    os.makedirs("results", exist_ok = True)
    climates = ['Bsh', 'Bsk', 'Bwh', 'Bwk', 'Cfa', 'Csa', 'Csb', 'Cwa', 'Dfa', 'Dfb', 'Dfc', 'Dsb', 'Dwb', 'Dwc', 'ET']
    for climate in climates:
        f = open("results/ConvCapsNet_" + climate + "_Accuracy.txt", "w")
        for run in range(10):
            model = TrainConvCapsNet(random_seed=run, climate=climate)
            accuracy = model.train()
            f.write(accuracy)
            f.write("\n")
        f.close()
