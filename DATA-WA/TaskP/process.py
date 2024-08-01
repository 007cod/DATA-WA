import torch
import torch.nn as nn
import numpy as np
import random
np.random.seed(42)     
def getdata(path, time_interval, max_num_task, grid_length = 0.5, gx=10, gy=10, is_train=True):
    tasks = [] 
    workers = []
    
    with open(path, 'r') as f:
        content = f.readlines()
        for d in content[1:]:
            d = d.split()
            if d[1] != "t":
                for _ in range(int(d[-3])):
                    workers.append(np.array([float(x) for x in d[0:1] + d[2:]]))
            else:
                tasks.append(np.array([float(x) for x in d[0:1] + d[2:]]))#[time, x, y, deadline, r]
    
    
    tasks = np.array(tasks)
    worker = np.array(workers)
    x_max = np.max(tasks[:, 1])
    y_max = np.max(tasks[:, 2])
    max_time = np.max(tasks[:, 0])
    
    if not is_train:
        return None, tasks, worker
    
    t_length = int((max_time + time_interval-1) // time_interval)
    begins = range(0, int(max_time-time_interval))
    
    lstm_tasks = np.zeros((len(begins), int(gx), int(gy), max_num_task), dtype=float)   # (t, gx, gy, max_num_task)
    time_dt = time_interval // max_num_task
    
    for i, b in enumerate(begins):
        for d in tasks:
            if d[0] >= b and d[0] < b + time_interval:
                lstm_tasks[i, int(d[1] // grid_length), int(d[2] // grid_length),  int((d[0]-b)//time_dt)] = 1
    
    # task_pos = np.zeros((len(begins), int(gx), int(gy), 1 , 2),dtype=float) 
    # for x in range(gx):
    #     for y in range(gy):
    #         task_pos[:,x,y,0,0] = x / gx
    #         task_pos[:,x,y,0,1] = y / gy
        
    return lstm_tasks, tasks, worker

def lstm_task_train_process(tasks:np.ndarray, time_interval, seq, train_ration = 0.7, train_num = 3000, device='cpu'):
    t_length = tasks.shape[0]
    tasks = tasks.reshape(t_length, -1)
    dataset_x = []
    dataset_y = []
    for i in range(t_length - (seq + 1)*time_interval):
        dataset_x.append(tasks[i: i + seq*time_interval : time_interval])
        dataset_y.append(tasks[i + seq*time_interval : i + seq*time_interval + 1])

    dataset_x, dataset_y = np.array(dataset_x), np.array(dataset_y)  #(len, seq, feature_size)
    # 随机打乱数据集
    ranidx = list(random.sample(range(0, len(dataset_x)), k=min(train_num,len(dataset_x))))
    dataset_x = dataset_x[ranidx]
    dataset_y = dataset_y[ranidx]
    
    train_size = int(len(dataset_x) * train_ration)
    train_x, train_y = torch.from_numpy(dataset_x[:train_size]).float(), torch.from_numpy(dataset_y[:train_size]).float()
    test_x, test_y = torch.from_numpy(dataset_x[train_size:]).float(), torch.from_numpy(dataset_y[train_size:]).float()
    
    train_x = train_x.permute(1,0,2).contiguous().to(device)
    train_y = train_y.permute(1,0,2).contiguous().to(device)
    test_x = test_x.permute(1,0,2).contiguous().to(device)
    test_y = test_y.permute(1,0,2).contiguous().to(device)
    return train_x, train_y, test_x, test_y  #(seq, len, feature_size)

def task_test_process(tasks:np.ndarray, now, time_interval, max_num_task, seq, model="LSTM",  gx = 10, gy = 10, grid_length = 0.5, device='cpu'):
    if len(tasks) <=0 or now - tasks[0][0] <= time_interval * seq:
        return None
    lstm_tasks = np.zeros((int(seq), int(gx), int(gy), int(max_num_task)),dtype=float)   # (seq_len, gx, gy, max_num_task, {t, p})
    begin = now - time_interval * seq + 1
    time_dt = time_interval//max_num_task
    temp_task = []
    
    for d in reversed(tasks):
        if now - d[0] >= time_interval * seq:
            break
        else:
            temp_task.append(d)
    
    for d in reversed(temp_task):
        lstm_tasks[int((d[0] - begin)//time_interval), int(d[1] // grid_length), int(d[2] // grid_length), int(((d[0] - begin) % time_interval)//time_dt)] = 1
    if model == "LSTM":
        lstm_tasks = lstm_tasks.reshape(1, seq, -1)
        
        lstm_tasks = torch.from_numpy(lstm_tasks).float()
        lstm_tasks = lstm_tasks.permute(1,0,2).contiguous()
    else:
        lstm_tasks = lstm_tasks.reshape(1, seq, -1, int(max_num_task))
        lstm_tasks = torch.from_numpy(lstm_tasks).float()
        lstm_tasks = lstm_tasks.permute(0, 3, 2, 1).contiguous()
    return lstm_tasks.to(device)
    
def gnn_task_train_process(tasks:np.ndarray, time_interval, max_num_task, seq , train_ration = 0.7, train_num = 3000, device='cpu'):
    t = len(tasks)
    tasks = tasks.reshape(int(t), -1, int(max_num_task))   #(t, num_node, feature_size)
    dataset_x = []
    dataset_y = []
    for i in range(t - (seq + 1)*time_interval):
        dataset_x.append(tasks[i: i + seq*time_interval : time_interval])
        dataset_y.append(tasks[i + seq*time_interval : i + seq*time_interval + 1])
        
    # 随机打乱数据集
    dataset_x, dataset_y = np.array(dataset_x), np.array(dataset_y)  #(len, seq, num_node, feature_size)
    ranidx = list(random.sample(range(0, len(dataset_x)), k=min(train_num,len(dataset_x))))
    dataset_x = dataset_x[ranidx]
    dataset_y = dataset_y[ranidx]
    
    train_size = int(len(dataset_x) * train_ration)
    train_x, train_y = torch.from_numpy(dataset_x[:train_size]).float(), torch.from_numpy(dataset_y[:train_size]).float()
    test_x, test_y = torch.from_numpy(dataset_x[train_size:]).float(), torch.from_numpy(dataset_y[train_size:]).float()
    
    train_x = train_x.permute(0, 3, 2, 1).contiguous().to(device)
    train_y = train_y.permute(0, 3, 2, 1).contiguous().to(device)
    test_x = test_x.permute(0, 3, 2, 1).contiguous().to(device)
    test_y = test_y.permute(0, 3, 2, 1).contiguous().to(device)
    
    return train_x, train_y, test_x, test_y  #(len, feature_size, num_node, seq) 

def grid_predict_to_task(predict, now, time_interval, max_num_task, threshold = 0.5, model="LSTM", gx = 10, gy = 10):
    if model == "LSTM":
        predict = predict.reshape(gx, gy, max_num_task)
        
    else:
        predict = predict.permute(0, 3, 2, 1).contiguous()
        predict = predict.reshape(gx, gy, int(max_num_task))
    predict = np.array(predict.cpu())
    tasks = [[[] for j in range(gy)] for i in range(gx)]
    
    for i in range(gx):
        for j in range(gy):
            for k in range(max_num_task):
                p = predict[i,j,k]
                if p > threshold:
                    tasks[i][j].append([p, time_interval * (k+0.5) / max_num_task + now + 1])
    return tasks
