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
from numpy import prod

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
    
def squash(s, dim=-1):
    '''
    "Squashing" non-linearity that shrunks short vectors to almost zero length and long vectors to a length slightly below 1
    Eq. (1): v_j = ||s_j||^2 / (1 + ||s_j||^2) * s_j / ||s_j||
    Args:
        s: Vector before activation
        dim: Dimension along which to calculate the norm
    Returns:
        Squashed vector
    '''
    squared_norm = torch.sum(s**2, dim=dim, keepdim=True)
    return squared_norm / (1 + squared_norm) * s / (torch.sqrt(squared_norm) + 1e-8)


class PrimaryCapsules(nn.Module):
    def __init__(self, in_channels, out_channels, dim_caps,
    kernel_size=5, stride=2, padding=0):
        """
        Initialize the layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dim_caps: Dimensionality, i.e. length, of the output capsule vector.
        """
        super(PrimaryCapsules, self).__init__()
        self.dim_caps = dim_caps
        self._caps_channel = int(out_channels / dim_caps)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), self._caps_channel, out.size(2), out.size(3), self.dim_caps)
        out = out.view(out.size(0), -1, self.dim_caps)
        return squash(out)


class RoutingCapsules(nn.Module):
    def __init__(self, in_dim, in_caps, num_caps, dim_caps, num_routing, device: torch.device):
        """
        Initialize the layer.

        Args:
            in_dim: Dimensionality (i.e. length) of each capsule vector.
            in_caps: Number of input capsules if digits layer.
            num_caps: Number of capsules in the capsule layer
            dim_caps: Dimensionality, i.e. length, of the output capsule vector.
            num_routing: Number of iterations during routing algorithm
        """
        super(RoutingCapsules, self).__init__()
        self.in_dim = in_dim
        self.in_caps = in_caps
        self.num_caps = num_caps
        self.dim_caps = dim_caps
        self.num_routing = num_routing
        self.device = device

        self.W = nn.Parameter( 0.01 * torch.randn(1, num_caps, in_caps, dim_caps, in_dim ) )

    def __repr__(self):
        tab = '  '
        line = '\n'
        next = ' -> '
        res = self.__class__.__name__ + '('
        res = res + line + tab + '(' + str(0) + '): ' + 'CapsuleLinear('
        res = res + str(self.in_dim) + ', ' + str(self.dim_caps) + ')'
        res = res + line + tab + '(' + str(1) + '): ' + 'Routing('
        res = res + 'num_routing=' + str(self.num_routing) + ')'
        res = res + line + ')'
        return res

    def forward(self, x):
        batch_size = x.size(0)
        # (batch_size, in_caps, in_dim) -> (batch_size, 1, in_caps, in_dim, 1)
        x = x.unsqueeze(1).unsqueeze(4)
        #
        # W @ x =
        # (1, num_caps, in_caps, dim_caps, in_dim) @ (batch_size, 1, in_caps, in_dim, 1) =
        # (batch_size, num_caps, in_caps, dim_caps, 1)
        u_hat = torch.matmul(self.W, x)
        # (batch_size, num_caps, in_caps, dim_caps)
        u_hat = u_hat.squeeze(-1)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()

        '''
        Procedure 1: Routing algorithm
        '''
        b = torch.zeros(batch_size, self.num_caps, self.in_caps, 1).to(self.device)

        for route_iter in range(self.num_routing-1):
            # (batch_size, num_caps, in_caps, 1) -> Softmax along num_caps
            c = F.softmax(b, dim=1)

            # element-wise multiplication
            # (batch_size, num_caps, in_caps, 1) * (batch_size, in_caps, num_caps, dim_caps) ->
            # (batch_size, num_caps, in_caps, dim_caps) sum across in_caps ->
            # (batch_size, num_caps, dim_caps)
            s = (c * temp_u_hat).sum(dim=2)
            # apply "squashing" non-linearity along dim_caps
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_caps, in_caps, dim_caps) @ (batch_size, num_caps, dim_caps, 1)
            # -> (batch_size, num_caps, in_caps, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv

        # last iteration is done on the original u_hat, without the routing weights update
        c = F.softmax(b, dim=1)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)

        return v
    
class CapsuleNetwork(nn.Module):
    def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, num_routing, device: torch.device, kernel_size=5):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.device = device

        self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.primary = PrimaryCapsules(channels, channels, primary_dim, kernel_size)
        
        primary_caps = int(channels / primary_dim * ( img_shape[1] - 2*(kernel_size-1) ) * ( img_shape[2] - 2*(kernel_size-1) ) / 4)
        self.digits = RoutingCapsules(primary_dim, primary_caps, num_classes, out_dim, num_routing, device=self.device)

        
#         self.decoder_et = nn.Sequential(
#             nn.Linear(num_classes, 1),
#         )

        self.metadata_network = torch.nn.Sequential(
            torch.nn.Linear(18, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64)
        )
        
        self.decoder_et = nn.Sequential(
            nn.Linear(((out_dim * num_classes) + 64), 1),
        )
        
        self.decoder_reconstruction = nn.Sequential(
            nn.Linear(out_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(prod(img_shape)) ),
            nn.Sigmoid()
        )

    def forward(self, x, metadata):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.primary(out)
        out = self.digits(out)
        preds = torch.norm(out, dim=-1)

        # Reconstruct the *predicted* image
        _, max_length_idx = preds.max(dim=1)
        y = torch.eye(self.num_classes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)

        reconstructions = self.decoder_reconstruction( (out*y).view(out.size(0), -1) )
#         reconstructions = self.decoder_reconstruction( (out*y).view(out.size(0), -1) )
        reconstructions = reconstructions.view(-1, *self.img_shape)
        
        
        y = self.metadata_network(metadata)
        et = self.decoder_et(torch.cat(((out).view(out.size(0), -1) , y), dim=1) )
#         et = self.decoder_et( (out*y).view(out.size(0), -1) )
        
#         et = self.decoder_et(preds)

        return et.flatten(), reconstructions
    
class CapsNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.Encoder = CapsuleNetwork(img_shape=(11, 32, 32), channels=256, primary_dim=16, num_classes=20, out_dim=16, num_routing=5, device=torch.device("cuda"))
        
#     seq_len, batch, input_size
    def forward(self, x, y):
        pred, reconstruction = self.Encoder(x, y)
        return pred, reconstruction
    
    
class TrainCapsNet():

    def __init__(self, epochs=300, batch_size=16, torch_type=torch.float32, random_seed=18, climate=None):
        super(TrainCapsNet, self).__init__()
        
        torch.cuda.manual_seed(18)
        torch.cuda.manual_seed_all(18)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        self.torch_type = torch_type
        
        self.mse = torch.nn.MSELoss()
        self.reconstruction_loss = torch.nn.MSELoss(size_average=False)
        self.model = CapsNet().to(self.device, dtype=torch.float32)
        
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
        
#         self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        
        self.opt = torch.optim.Adagrad(self.model.parameters(), lr_decay=0.01)
        #self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.9)
        #self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.9)
    
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
                output, reconstruction = self.model(img_seq, torch.cat((veg, day_sin, day_cos, x, y, z, elev), dim=1))
                loss = self.mse(output, et) + (5e-4 * self.reconstruction_loss(reconstruction, img_seq))
                loss.backward()
                self.opt.step()
            #self.sched.step()
            self.accuracy = self.test(epoch)
        torch.save(self.model, "checkpoints/CapsNet_" + climate + "_" + str(self.random_seed) + ".pt" )
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
                output, _ = self.model(img_seq, torch.cat((veg, day_sin, day_cos, x, y, z, elev), dim=1))
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
        f = open("results/CapsNet_" + climate + "_Accuracy.txt", "w")
        for run in range(10):
            model = TrainCapsNet(random_seed=run, climate=climate)
            accuracy = model.train()
            f.write(accuracy)
            f.write("\n")
        f.close()
