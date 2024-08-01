import os 
from tqdm.std import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import numpy as np
from TaskP.module import Lstm_G, GNN_G, GNN_G4
from TaskP.process import getdata
from task_assign import Task_Assignment
from Qmodel import Qnet
from utils import qdata_process

arg = {
    "time_interval":100,
    "seq":10,    #序列大小
    "gx":10,    #x轴格点数量
    "gy":10,
    "max_num_task":20,  #模型所能预测单个格点的最大任务数量
    "grid_length":0.5,  #格点一个格点对应的实际长度
    "nw":10,
    "nt":20,
    "nq":4,
    "maxRS":5,      #RS的最大数量
    "maxPVTS":2,
    "is_greedy":False,
    "fpta":False,
    "threshold":0.85,  #任务预测的阈值
    "is_predict":False,
    "task_model":"GNN",
    "is_train_RL":True,
    "is_bfs":False,
    "data_path" : "./dataset/dd/dd_21_23_train.txt" ,  #"./dataset/gMission/gMission.txt"  "./dataset/synthetic/synthetic.txt" 
    "model_weight_path" : "./TaskP/output/GNN4_best_dd_100_21_23.pth",  #"./TaskP/output/GNN4_best_yc2_1002.pth"
    "qmodel_path": "./RL_model/RL_dd_best_60_52.pth",
    "device":'cuda' if torch.cuda.is_available() else 'cpu'
}

if __name__ == "__main__":
    #初始化数据
    lstm_task, tasks, workers = getdata(arg["data_path"], time_interval=arg["time_interval"], max_num_task=arg["max_num_task"])
    #train_x, train_y, test_x, test_y = task_train_process(lstm_task, seq=arg["seq"]) #(seq, len, feature_size)
    
    #feature_size = train_x.shape[-1]
    max_time = max(np.max(tasks[:,0]), np.max(workers[:,0]))
    
    taskModel = None
    if arg["is_predict"]:   
        if arg["task_model"] == "LSTM":
            #taskModel = Lstm_G(feature_size)
            pass
        else:
            num_nodes = arg['gx'] * arg['gy']
            taskModel = GNN_G4(num_nodes, arg['max_num_task'], arg['max_num_task'], hidden_size=256, num_layer=4)

        taskModel.load_state_dict(torch.load(arg["model_weight_path"],map_location=torch.device(arg['device'])))
    
    Qmodel = None
    if arg['is_bfs']:
        Qmodel =  Qnet(arg["nt"], arg["nw"], arg["nq"], hidden_size=60)
        Qmodel.load_state_dict(torch.load(arg["qmodel_path"],map_location=torch.device(arg['device'])))
        
    ti = 0
    wi = 0
    R = 0
    
    LPTA = Task_Assignment(is_bfs=arg["is_bfs"],is_predict=arg["is_predict"], Pmodel=taskModel, is_train_RL=arg['is_train_RL'], arg=arg)
    
    tasks_data = []
    for i in range(0, int(max_time+1000)):
        newtasks = []
        newworkers = []
        while ti < len(tasks) and tasks[ti][0] <= i:
            newtasks.append(tasks[ti])
            tasks_data.append(tasks[ti])
            ti += 1
        while wi<len(workers) and workers[wi][0] <= i:
            newworkers.append(workers[wi])
            wi += 1
        is_task_ass = LPTA.add_new(newworkers, newtasks, i)
        if is_task_ass or LPTA.next_time < i:
            LPTA.init(i)
            if arg["is_predict"]:
                LPTA.fresh_SP(tasks_data, i)   #重新进行任务预测
            LPTA.assign(i)
            LPTA.init(i, del_pred=False)
        LPTA.init(i, del_pred = False)
        
    LPTA.init(max_time+1000)
    print(f'R:{LPTA.R}')
    
    model = Qnet(arg["nt"], arg["nw"], arg['nq'], hidden_size=60)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    Qdata, r = qdata_process(LPTA.memory, nt=arg["nt"], nw=arg["nw"], nq=arg["nq"])
    train_num = Qdata[0].shape[0]
    
    batch = 64
    best_loss = 1
    train_loss = []
    for i in range(400):
        total_loss = 0
        for j in tqdm(range(0, train_num, batch), postfix=total_loss):
            end = j + batch if j + batch <= train_num else train_num
            out = model(Qdata[0][j:end], Qdata[1][j:end],Qdata[2][j:end],Qdata[3][j:end])
            loss = loss_function(out[:,0], r[j:end])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()*batch/train_num
        train_loss.append(total_loss)
        # 将训练过程的损失值写入文档保存，并在终端打印出来
        if total_loss < best_loss:
            torch.save(model.state_dict(), arg["qmodel_path"])
            best_loss = total_loss
        with open('log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i+1, total_loss))
        if (i+1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, total_loss))