import torch
import torch.nn as nn
import os
from module import *
from waveNet import gwnet
import numpy as np
from tqdm.std import tqdm
from process import getdata, lstm_task_train_process, gnn_task_train_process
from util import get_acc_num, gnn_loss, FocalLoss, get_PR
from process import grid_predict_to_task
import random
import copy
import time

def train_main(arg):
    #数据处理
    Lstm_tasks, tasks, workers = getdata(arg["data_path"], time_interval=arg["time_interval"], max_num_task=arg["max_num_task"])
    
    if arg["model"] == "LSTM":
        train_x, train_y, test_x, test_y = lstm_task_train_process(Lstm_tasks, time_interval=arg["time_interval"], train_ration=arg["train_ration"], seq=arg["seq"], train_num=arg["train_num"], device=arg['device'])
        seq, train_num, feature_size = train_x.shape
        model = Lstm_G(input_size=feature_size)
    elif arg["model"] == "GNN":
        train_x, train_y, test_x, test_y = gnn_task_train_process(Lstm_tasks, time_interval=arg["time_interval"], train_ration=arg["train_ration"], seq=arg["seq"], max_num_task=arg["max_num_task"], train_num=arg["train_num"], device=arg['device'])  #归一化和Tensor转换
        train_num, feature_size, num_node, seq = train_x.shape
        model = GNN_G4(num_nodes=num_node, input_size=feature_size, output_size=feature_size, hidden_size=arg["hidden_size"], num_layer=arg["num_layer"])
    else:
        train_x, train_y, test_x, test_y = gnn_task_train_process(Lstm_tasks, time_interval=arg["time_interval"], train_ration=arg["train_ration"], seq=arg["seq"], max_num_task=arg["max_num_task"], train_num=arg["train_num"], device=arg['device'])  #归一化和Tensor转换
        train_num, feature_size, num_node, seq = train_x.shape
        model = gwnet('cpu', num_nodes=num_node, in_dim=feature_size, out_dim=feature_size, layers=4)
    
    #model.load_state_dict(torch.load(arg["output_path"] + "_" + str(arg["time_interval"]) +  ".pth",map_location=torch.device(arg['device'])))
    #model = gwnet('cpu', num_nodes=num_node, in_dim=20, out_dim=20, layers=4)
    tt1 = time.time()
    model.to(arg['device'])
    model_total = sum([param.nelement() for param in model.parameters()]) # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total/1e6))
    
    train_loss = []
    loss_function = FocalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=arg["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    best_loss = 1
    batch = arg["batch"]
    log_path = arg["output_path"] + "_" + str(arg["time_interval"]) + "_log.txt"
    for i in range(1, arg["epoch"]):
        total_loss = 0
        for j in tqdm(range(0, train_num, batch), postfix=total_loss):
            end = j + batch if j + batch <= train_num else train_num
            if arg["model"] == "LSTM":
                out = model(train_x[:,j:end,:])
                loss = loss_function(out, train_y[:,j:end,:])
            else:
                out = model(train_x[j:end])
                loss = loss_function(out, train_y[j:end])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()/((train_num + batch - 1)//batch)
        train_loss.append(total_loss)
        
        with open(log_path, 'a+') as f:
            f.write('{} - {}\n'.format(i+1, total_loss))
        if (i+1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, total_loss))
        
        if i % 1 == 0:
            print(f"val: {(time.time() - tt1)*200/3600}")
            with torch.no_grad():
                if arg["model"] == "LSTM":
                    test_num = test_x.shape[1]
                else:
                    test_num = test_x.shape[0]
                test_loss = 0
                acc = 0
                err = 0
                test_task_num = 0
                all_out = copy.deepcopy(test_y)
                ans_t1 = 0
                for j in range(0, test_num):
                    
                    if arg["model"] == "LSTM":
                        tempx = test_x[:,j:j+1,:]
                        tempy = test_y[:,j:j+1,:]
                    else:
                        tempx = test_x[j:j+1]
                        tempy = test_y[j:j+1]
                    t1 = time.time()
                    out = model(tempx)
                    ans_t1 += (time.time() - t1)/test_num
                    
                    if arg["model"] == "LSTM":
                        all_out[:,j:j+1,:] = out 
                    else:
                        all_out[j:j+1] = out
                    mask = tempy > 0
                    loss = loss_function(out, tempy)
                    tacc, terr= get_acc_num(out, tempy, time_interval=arg["time_interval"], max_num_task=arg["max_num_task"], model=arg["model"])
                    
                    acc += tacc
                    err += terr
                    test_task_num += torch.sum(mask)
                    test_loss += loss.item()/test_num
                # 将训练过程的损失值写入文档保存，并在终端打印出来
                print(ans_t1)
                AP = get_PR(all_out, test_y, only_print=True)
                if test_loss < best_loss:
                    torch.save(model.state_dict(), arg["output_path"] + "_" + str(arg["time_interval"]) +  ".pth")
                    best_loss = test_loss
                print('Epoch: {}, val_Loss:{:.5f}'.format(i+1, test_loss))
                print(f"ar:{acc/(test_task_num + 1)}, ap:{acc/(acc + err + 1)}, AP:{AP}")
                with open(log_path, 'a+') as f:
                    f.write(f"ar:{acc/(test_task_num + 1)}, ap:{acc/(acc + err + 1)}, AP:{AP}\n")
                    

if __name__ == "__main__":
    
    arg = {
        "data_path" : "./dataset/dd/dd_21_23_train.txt",# ./dataset/synthetic/synthetic.txt  ./dataset/gMission/gMission.txt
        "output_path":'./TaskP/output/temp/gwnet_best_dd',
        "model":"GNN",
        "train_num":2500,
        "train_ration":0.8,
        "batch":32,
        "epoch":2,
        "lr":0.001,
        
        "time_interval":140,
        "max_num_task":20,
        "seq":10,
        
        "hidden_size":256,
        "num_layer":4,
        
        "device":'cuda' if torch.cuda.is_available() else 'cpu'
    }
    for num in list([100, 120, 140, 160, 180]):
        print(num)
        arg["time_interval"] = num
        train_main(arg)
    