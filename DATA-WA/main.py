import os
import time
import argparse
from tqdm.std import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
random.seed(32)

from TaskP.module import Lstm_G, GNN_G, GNN_G3, GNN_G4
from TaskP.waveNet import gwnet
from TaskP.process import getdata
from task_assign import Task_Assignment
from utils import action_vis
from Qmodel import Qnet

def parse_args():
    parser = argparse.ArgumentParser(description="Task Assignment Parameters")
    parser.add_argument("--time_interval", type=int, default=100)
    parser.add_argument("--seq", type=int, default=10)
    parser.add_argument("--gx", type=int, default=10)
    parser.add_argument("--gy", type=int, default=10)
    parser.add_argument("--max_num_task", type=int, default=20)
    parser.add_argument("--grid_length", type=float, default=0.5)
    parser.add_argument("--nw", type=int, default=20)
    parser.add_argument("--nt", type=int, default=20)
    parser.add_argument("--nq", type=int, default=5)
    parser.add_argument("--maxRS", type=int, default=4)
    parser.add_argument("--maxPVTS", type=int, default=2)
    parser.add_argument("--is_greedy", action='store_true')
    parser.add_argument("--fpta", action='store_true')
    parser.add_argument("--is_predict", action='store_true')
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--task_model", type=str, default="GNN")
    parser.add_argument("--is_train_RL", action='store_true')
    parser.add_argument("--is_bfs", action='store_true')
    parser.add_argument("--data_path", type=str, default="./dataset/dd/dd.txt")
    parser.add_argument("--model_weight_path", type=str, default="./TaskP/output/GNN4_best_dd_1003.pth")
    parser.add_argument("--qmodel_path", type=str, default="./RL_model/RL_dd_best.pth")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--worker_num", type=int, default=500)
    parser.add_argument("--task_num", type=int, default=10959)
    parser.add_argument("--reach_distance", type=float, default=57/111)
    parser.add_argument("--deadline", type=int, default=40)
    parser.add_argument("--wd", type=float, default=1)
    parser.add_argument("--log_path", type=str, default="./log.txt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 初始化数据
    lstm_task, tasks, workers = getdata(args.data_path, time_interval=args.time_interval, max_num_task=args.max_num_task)
    max_time = max(np.max(tasks[:,0]), np.max(workers[:,0]))

    # 初始化模型
    taskModel = None
    if args.is_predict:   
        if args.task_model == "LSTM":
            taskModel = Lstm_G(input_size=100*args.max_num_task)
        elif args.task_model == "GNN":
            num_nodes = args.gx * args.gy
            taskModel = GNN_G4(num_nodes, args.max_num_task, args.max_num_task, hidden_size=256, num_layer=4)
        elif args.task_model == "gwnet":
            num_nodes = args.gx * args.gy
            taskModel = gwnet('cpu', num_nodes=num_nodes, in_dim=args.max_num_task, out_dim=args.max_num_task, layers=4)
            
        taskModel.load_state_dict(torch.load(args.model_weight_path, map_location=torch.device(args.device)))
        taskModel.to(args.device)
    
    Qmodel = None
    if args.is_bfs:
        Qmodel = Qnet(args.nt, args.nw, args.nq, hidden_size=60)
        Qmodel.load_state_dict(torch.load(args.qmodel_path, map_location=torch.device(args.device)))
        Qmodel.to(args.device)
    ti = 0
    wi = 0
    R = 0
    print(args)
    LPTA = Task_Assignment(is_bfs=args.is_bfs, is_predict=args.is_predict, Qmodel=Qmodel, Pmodel=taskModel, is_train_RL=False, arg=vars(args))
    vis_datas = []
    reward = []
    
    assign_time = []
    tasks_data = []
    worker_idx = list(random.sample(range(0, len(workers)), args.worker_num if args.worker_num != -1 else len(workers)))
    worker_idx.sort()
    workers = workers[worker_idx]
    task_idx = list(random.sample(range(0, len(tasks)), args.task_num if args.task_num != -1 else len(tasks)))
    task_idx.sort()
    tasks = tasks[task_idx]
    
    for i in range(0, int(max_time + 1000)):
        newtasks = []
        newworkers = []
        while ti < len(tasks) and tasks[ti][0] <= i:
            tasks_data.append(tasks[ti])
            # if tasks[ti][1] > 2.5 or tasks[ti][2] > 2.5:
            #     ti += 1
            #     continue
            newtasks.append(tasks[ti])
            ti += 1
        while wi < len(workers) and workers[wi][0] <= i:
            # if workers[wi][1] > 2.5 or workers[wi][2] > 2.5:
            #     wi += 1
            #     continue
            newworkers.append(workers[wi])
            wi += 1
        is_task_ass = LPTA.add_new(newworkers, newtasks, i)
        if is_task_ass or LPTA.next_time <= i:
            LPTA.init(i)
            t1 = time.time()
            if args.is_predict:
                LPTA.fresh_SP(tasks_data, i)   # 重新进行任务预测
            LPTA.assign(i)
            LPTA.init(i, del_pred=False)
            if i > 1000:
                assign_time.append(time.time() - t1)
        LPTA.init(i, del_pred=False)
        vis_datas.append(LPTA.get_vis_data(i))
        reward.append(LPTA.R)
    LPTA.init(max_time + 1000)
    print(f'R:{LPTA.R}, acc:{LPTA.acc}, avgTime:{sum(assign_time) / len(assign_time)}')
    with open(args.log_path, "+a") as f:
        f.write(f'reach_distance:{args.reach_distance / 0.51:.2f}, deadline:{args.deadline}, R:{LPTA.R}, acc:{LPTA.acc}, avgTime:{sum(assign_time) / len(assign_time):.7f}, Greedy:{args.is_greedy}, FPTA:{args.fpta}, predict:{args.is_predict}, BFS:{args.is_bfs}, wd:{args.wd}, tl:{args.time_interval}, model:{args.task_model}\n')
        #f.write(f'task_num:{args.task_num}, worker_num:{args.worker_num}, R:{LPTA.R}, acc:{LPTA.acc}, avgTime:{sum(assign_time) / len(assign_time):.7f}, Greedy:{args.is_greedy}, FPTA:{args.fpta}, predict:{args.is_predict}, BFS:{args.is_bfs} \n')
    # action_vis(vis_datas, reward)
