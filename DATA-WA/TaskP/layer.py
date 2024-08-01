import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GCNConv

#用于裁剪输入张量的时间维度，去除多余的 padding 部分。
class Crop(nn.Module):
 
    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size
 
    def forward(self, x):
        #裁剪张量以去除额外的填充
        return x[:, :, :, :-self.crop_size].contiguous()

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,5]
        for kern in self.kernel_set:
            padding = (kern - 1) * dilation_factor
            self.tconv.append(nn.Sequential(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor), padding=(0, padding)), Crop(padding), nn.Dropout(0.2)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        x = sum(x)
        return x
    
class construct_adj(nn.Module):
    def __init__(self, nnodes, dim, alpha=3, static_feat=None):
        super(construct_adj, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat
    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0)) + torch.mm(nodevec2, nodevec1.transpose(1,0))
        a = torch.softmax(a, dim=-1)
        adj = self.alpha*a
        return adj

class construct_adj4(nn.Module):
    def __init__(self, nnodes, dim):
        super(construct_adj4, self).__init__()
        self.nnodes = nnodes
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.dim = dim
    def forward(self, x: torch.Tensor):  #(b, c, n, s)
        x1, x2 = torch.tanh(self.conv1(x)), torch.tanh(self.conv2(x))
        
        a = torch.einsum('bcns,bcks->bnks',(x1, x2))
        a.contiguous()

        adj = torch.softmax(a, dim=-2)

        return adj

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class nconv4(nn.Module):
    def __init__(self):
        super(nconv4,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('bcns,bvns->bcvs',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vwl->ncvl',(x,A))
        return x.contiguous()

class prop(nn.Module):
    def __init__(self,c_in,c_out,k,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.k = k
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.k):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho
    
class prop4(nn.Module):
    def __init__(self,c_in,c_out,k,alpha):
        super(prop4, self).__init__()
        self.nconv = nconv4()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1))
        self.k = k
        self.alpha = alpha

    def forward(self, x, adj):
        b, n, n, s = adj.shape
        adj = adj + torch.eye(adj.size(1)).to(x.device).reshape(1, n, n, 1)
        d = adj.sum(2)
        h = x
        dv = d
        a = adj / dv.reshape(b, n, 1, s)
        for i in range(self.k):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho

class GCN_block3(nn.Module):
    def __init__(self, hidden_size = 20, k=1, alpha = 0.1):
        super(GCN_block3, self).__init__()
        self.filter = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.gate = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.prop = prop(c_in=hidden_size, c_out=hidden_size, k=k, alpha=alpha)
        #GCN()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, adj):
        filter = self.filter(x)
        filter = self.tanh(filter)
        gate = self.gate(x)
        
        gate = self.sigmoid(gate)
        out = filter * gate
        out = self.prop(out, adj)

        return out


class GCN_block(nn.Module):
    def __init__(self, hidden_size = 100, k=1, alpha = 0.1):
        super(GCN_block, self).__init__()
        self.filter = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.gate = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.ppnp = APPNP(K=k, alpha=alpha)
        #GCN()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_weight):
        filter = self.filter(x)
        filter = self.tanh(filter)
        gate = self.gate(x)
        
        gate = self.sigmoid(gate)
        out = filter * gate
        
        b,c,n,s = out.shape
        out = out.permute(0, 2, 1 ,3)
        
        for i in range(b):
            for j in range(s):
                out[i,:,:,j] = self.ppnp(out[i,:,:,j], edge_index, edge_weight)
        
        out = out.permute(0, 2, 1 ,3).contiguous()
        return out

class GCN_block2(nn.Module):
    def __init__(self, hidden_size = 100, k=10, alpha = 0.1):
        super(GCN_block2, self).__init__()
        self.filter = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.gate = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.gcn = GCNConv(in_channels=hidden_size, out_channels=hidden_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, edge_index, edge_weight):
        filter = self.filter(x)
        filter = self.tanh(filter)
        gate = self.gate(x)
        
        gate = self.sigmoid(gate)
        out = filter * gate
        
        b,c,n,s = out.shape
        
        out = out.permute(0, 2, 1 ,3).contiguous()
        out_temp = out.clone()
        
        for i in range(b):
            for j in range(s):
                out_temp[i,:,:,j] = self.gcn(out[i,:,:,j], edge_index, edge_weight)
        
        out_temp = out_temp.permute(0, 2, 1 ,3).contiguous()
        return out_temp

class GCN_block4(nn.Module):
    def __init__(self, hidden_size = 20, k=1, alpha = 0.1):
        super(GCN_block4, self).__init__()
        self.filter = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.gate = dilated_inception(cin=hidden_size, cout=hidden_size)
        self.prop = prop4(c_in=hidden_size, c_out=hidden_size, k=k, alpha=alpha)
        #GCN()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, adj):
        filter = self.filter(x)
        filter = self.tanh(filter)
        gate = self.gate(x)
        
        gate = self.sigmoid(gate)
        out = filter * gate
        out = self.prop(out, adj)

        return out