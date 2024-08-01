import torch
import torch.nn as nn
import numpy as np
from process import grid_predict_to_task
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def get_acc_num(out, tg, time_interval, max_num_task, model, threshold=0.5):   #(len, feature_size, num_node, seq)
    out = grid_predict_to_task(out, model=model, now=1, time_interval=time_interval, max_num_task=max_num_task)
    tg = grid_predict_to_task(tg, model=model, now=1, time_interval=time_interval, max_num_task=max_num_task)
    
    gx = len(out)
    gy = len(out[0])
    acc = 0
    err = 0
    for i in range(gx):
        for j in range(gy):
            for ip, p in enumerate(out[i][j]):
                if p[0] > threshold:
                    err += 1
                    for k, g in enumerate(tg[i][j]):
                        if g[0] == 1 and abs(g[1]-p[1]) < 5:
                            acc += 1
                            g[0] = 0
                            err -= 1
                            break
    
    return acc, err

def get_PR(out, tg, only_print=False):
    #out = out.permute(0, 3, 2, 1).contiguous()
    out = out.reshape(-1, )
    out = np.array(out.cpu())
    #tg = tg.permute(0, 3, 2, 1).contiguous()
    tg = tg.reshape(-1,)
    tg = np.array(tg.cpu())
    
    # 计算Precision-Recall曲线
    precision, recall, _ = precision_recall_curve(tg, out)
    average_precision = average_precision_score(tg, out)
    print(average_precision)
    if only_print:
        return average_precision
    # 绘制PR曲线
    plt.figure()
    plt.step(recall, precision, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:0.2f}')
    plt.show()


class gnn_loss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mse = nn.MSELoss(reduction="sum")
        self.msem = nn.MSELoss()
        
    def forward(self, pred, target):
        mask = target  == 1
        nemask =  target == 0
        loss1 = 0.6*(self.mse(pred * mask, target * mask))/(torch.sum(mask)+ 1) + 0.4*self.mse (pred * nemask, target * nemask)/(torch.sum(nemask) +1)
        #loss2 = self.msem(pred, target)
        return loss1
    
class FocalLoss(nn.Module):
	def __init__(self,alpha=0.9,gamma=1):
		super(FocalLoss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self,preds,labels):
		"""
		preds:sigmoid的输出结果
		labels：标签
		"""
		eps=1e-7
		loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)
