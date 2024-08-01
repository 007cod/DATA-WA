import torch
import torch.nn as nn
import os
import copy
from module import *
from waveNet import gwnet
import numpy as np
from tqdm.std import tqdm
from process import getdata, gnn_task_train_process, lstm_task_train_process
from util import get_acc_num, gnn_loss, FocalLoss, get_PR
from process import grid_predict_to_task

def test_main(arg):
    #数据处理
    Lstm_tasks, tasks, workers = getdata(arg["data_path"], time_interval=arg["time_interval"], max_num_task=arg["max_num_task"])
    
    if arg["model"] == "LSTM":
        train_x, train_y, test_x, test_y = lstm_task_train_process(Lstm_tasks, time_interval=arg["time_interval"], train_ration=arg["train_ration"], seq=arg["seq"], train_num=arg["train_num"], device=arg['device'])
        seq, train_num, feature_size = train_x.shape
        model = Lstm_G(input_size=feature_size)
    else:
        train_x, train_y, test_x, test_y = gnn_task_train_process(Lstm_tasks, time_interval=arg["time_interval"], train_ration=arg["train_ration"], seq=arg["seq"], max_num_task=arg["max_num_task"], train_num=arg["train_num"], device=arg['device'])  #归一化和Tensor转换
        train_num, feature_size, num_node, seq = train_x.shape
        model = GNN_G4(num_nodes=num_node, input_size=feature_size, output_size=feature_size, hidden_size=arg["hidden_size"], num_layer=arg["num_layer"])
    model.to(arg['device'])
    model_total = sum([param.nelement() for param in model.parameters()]) # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total/1e6))
    
    with torch.no_grad():
        if arg["model"] == "LSTM":
            seq, train_num, feature_size = test_x.shape
            out = copy.deepcopy(test_y)
            for j in tqdm(range(0, test_num)):
                temp_out = model(test_x[:,j:j+1,:])
                out[:,j:j+1,:] = temp_out
        else:
            test_num, feature_size, num_node, seq = test_x.shape
            out = copy.deepcopy(test_y)
            for j in tqdm(range(0, test_num)):
                temp_out = model(test_x[j:j+1])
                
                out[j:j+1,:,:,:] = temp_out
        get_PR(out, test_y)


if __name__ == "__main__":
    
    arg = {
        "data_path" : "./dataset/yc/yc_train.txt",# ./dataset/synthetic/synthetic.txt  ./dataset/gMission/gMission.txt
        "output_path":'./TaskP/output/GNN4_best_dd',
        "train_num":1800,
        "train_ration":0.8,
        "batch":128,
        "epoch":200,
        "lr":0.001,
        
        "time_interval":100,
        "max_num_task":10,
        "seq":10,
        
        "hidden_size":256,
        "num_layer":4,
        "threshold":0.5,
        "device":'cuda' if torch.cuda.is_available() else 'cpu'
    }
    #train_LSTM(data_path, train_ration, batch)
    test_main(arg)
    