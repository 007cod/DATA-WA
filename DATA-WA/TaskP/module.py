import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GCNConv
sys.path.append(".") 
from TaskP.layer import *

class Lstm_G(nn.Module):
    def __init__(self, input_size, hidden_size = 100, num_layer = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer)
        self.fc = nn.Linear(hidden_size, input_size)
        self.sigmod = nn.Sigmoid()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        s, b, h = x.shape
        out = self.fc(x[-1,:,:])
        out = self.sigmod(out)
        out = out.view(1,b, -1)
        return out
         
class GNN_G(nn.Module):
    def __init__(self, num_nodes, input_size, output_size, hidden_size = 20, num_layer = 4):
        super(GNN_G, self).__init__()
        self.firstConv = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.get_adj = construct_adj(num_nodes, hidden_size)
        
        self.layer = nn.ModuleList()
        for i in range(num_layer):
            self.layer.append(GCN_block2(hidden_size, k=1))
            
        self.relu = nn.LeakyReLU()
        self.lastConv = nn.Conv2d(hidden_size, output_size, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.sigmod = nn.Sigmoid()
        
        
        self.num_layer = num_layer
        self.num_nodes = num_nodes
        self.idx = torch.arange(self.num_nodes)
        
    def forward(self, x):
        
        x = self.firstConv(x)
        adj = self.get_adj(self.idx)
        
        edge_idx = torch.nonzero(adj > 0.2)
        
        edge_weight = adj[edge_idx[:,0], edge_idx[:, 1]]
        edge_idx = edge_idx.permute(1, 0)
        
        temp = [x,]
        out = x
        
        for i in range(self.num_layer):
            new = self.layer[i](temp[-1], edge_idx, edge_weight) + temp[-1]
            temp.append(new)
            out = out + new
        out = out[...,-1:]
        out = self.relu(out)
        out = self.lastConv(out) 
        out = self.sigmod(out)
        return out
    
class GNN_G3(nn.Module):
    def __init__(self, num_nodes, input_size, output_size, hidden_size = 20, num_layer = 4):
        super(GNN_G3, self).__init__()
        self.firstConv = nn.Conv2d(input_size, hidden_size, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.get_adj = construct_adj(num_nodes, hidden_size)
        
        self.layer = nn.ModuleList()
        for i in range(num_layer):
            self.layer.append(GCN_block3(hidden_size, k=1))
            
        self.relu = nn.LeakyReLU()
        self.lastConv = nn.Conv2d(hidden_size, output_size, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.sigmod = nn.Sigmoid()
        
        
        self.num_layer = num_layer
        self.num_nodes = num_nodes
        self.idx = torch.arange(self.num_nodes)
        
    def forward(self, x):
        
        x = self.firstConv(x)
        adj = self.get_adj(self.idx)
        temp = [x,]
        out = x
        
        for i in range(self.num_layer):
            new = self.layer[i](temp[-1], adj) + temp[-1]
            temp.append(new)
            out = out + new
        out = out[...,-1:]
        out = self.relu(out)
        out = self.lastConv(out) 
        out = self.sigmod(out)
        return out

class GNN_G4(nn.Module):
    def __init__(self, num_nodes, input_size, output_size, hidden_size = 20, num_layer = 4):
        super(GNN_G4, self).__init__()
        self.firstConv = nn.Sequential()
        self.firstConv.append(nn.Conv2d(input_size, hidden_size//4, kernel_size=(1, 1), padding=(0,0), stride=(1,1)))
        self.firstConv.append(nn.Conv2d(hidden_size//4, hidden_size//2, kernel_size=(1, 1), padding=(0,0), stride=(1,1)))
        self.firstConv.append(nn.Conv2d(hidden_size//2, hidden_size, kernel_size=(1, 1), padding=(0,0), stride=(1,1)))
        self.get_adj = construct_adj4(num_nodes, hidden_size)
        
        self.layer = nn.ModuleList()
        for i in range(num_layer):
            self.layer.append(GCN_block4(hidden_size, k=1))
            
        self.relu = nn.LeakyReLU()
        self.lastConv = nn.Conv2d(hidden_size, output_size, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.sigmod = nn.Sigmoid()
            
        
        self.num_layer = num_layer
        self.num_nodes = num_nodes
        
    def forward(self, x):
        
        x = self.firstConv(x)
        adj = self.get_adj(x)
        temp = [x,]
        out = x
        
        for i in range(self.num_layer):
            new = self.layer[i](temp[-1], adj) + temp[-1]
            temp.append(new)
            out = out + new
        out = out[...,-1:]
        out = self.relu(out)
        out = self.lastConv(out) 
        out = self.sigmod(out)
        return out