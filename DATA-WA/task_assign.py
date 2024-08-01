import os
import math
import copy
import time
import numpy as np
import torch
from utils import qdata_process, plot_tree
from TaskP.process import task_test_process, grid_predict_to_task

class Task():
    def __init__(self, id, task, deadline, predicted=1) -> None:
        self.id = id
        self.t = task[0]
        self.x = task[1]
        self.y = task[2]
        self.d = deadline
        self.r = task[4]
        self.w = -1
        self.predicted = predicted   #任务预测概率
        self.state = 0  #工人状态0：空闲，1：正在被工人工作

class Worker():
    def __init__(self, id, worker, reach_distance, wd) -> None:
        self.id = id
        self.t = worker[0]
        self.x = worker[1]
        self.y = worker[2] 
        self.r = reach_distance #worker[3]  sy:2.5 dd:57/111
        self.c = worker[4]
        self.d = wd * 3600  #sy:300 dd:1800
        self.s = worker[6]
        self.v = 57*40/111/3600  #速度  sy:3/111  dd:57*30/111/3600
        self.state = 0   #工人状态0：空闲，1：工作
        
        self.dx = 0  #方向
        self.dy = 0
        self.lastTPS = []  #已完成的任务序列 
        self.TPS = []     #任务规划序列 对应self.S中的索引， id小于0的我们认为分配的是预测任务  
        
        self.RS = []     #可达任务集合
        self.idle_time = worker[0]  #预计空闲时刻(完成第一个任务的时刻)
        self.finish_time = 0 #完成所有任务的预计时刻

class Task_Assignment():
    def __init__(self, is_bfs = False, is_predict=False, Qmodel = None, Pmodel = None, is_train_RL=False, arg = None) -> None:
        self.max_time = 100   #窗口时间间隔
        self.max_x = 5
        self.max_y = 5
        self.arg = arg
        
        self.ttt=0
        self.acc = 0
        self.t = 0    #当前时间
        self.R = 0    #总利润
        self.all_num_w = 0
        self.all_num_s = 0
        self.next_time = 1000000000   #最早出现空闲工人的时间点
        
        self.is_greedy = arg["is_greedy"]
        self.is_bfs = is_bfs  #采用遍历的方式
        self.Qmodel = Qmodel  #价值评估模型
        self.is_predict = is_predict  #是否采用任务预测
        self.is_train_RL = is_train_RL #是否训练价值函数
        self.Pmodel = Pmodel
        self.sp_idx = 0
        
        self.W = []   #(id, t, x, y, r, d, s)
        self.S = []   #(id, t, x, y, d, r)
        
        self.Widx = []  #正在工作的工人对应的索引
        self.Sidx = []
        
        self.num_w = 0
        self.num_s = 0
        self.MPVST = []
        
        self.e = None
        self.ne = None
        self.h = None
        self.idx = 0
        
        self.root_node = []  #多棵树的根节点
        self.tree = None   #树节点存储的集合是self.Widx的索引
        self.tree_children = None
        self.tree_all_children_node = None  #该树节点下所有子节点的工人集合的并集
        self.idxtree = 0
        self.up_num = []   #树上每个节点的最大完成任务数量
        
        self.memory = []   #RL训练数据[(state, action, reward),]
        self.memorylen = 0
        self.max_mem_len= 10000
        
        self.SPgrid = [[[] for j in range(arg['gy'])] for i in range(arg['gx'])]   #(gx, gy)  代表着当前的预测任务
        self.SP = []
    
    def init(self, t, del_pred=True):
        #移除已经完成任务的工人和任务
        self.t = t
        self.next_time = 1000000000
        for wi in copy.deepcopy(self.Widx):
            w = self.W[wi]
            if w.idle_time <= t and len(w.TPS) >= 1 and w.state == 1:  #此时工人正好完成一个任务
                w.lastTPS.append(w.TPS[0])
                w.x, w.y = self.S[w.TPS[0]].x, self.S[w.TPS[0]].y   #更新工人位置
                self.R += 1
                self.Sidx.remove(w.TPS[0])  #去除已经完成的任务
                w.TPS.pop(0)
                
                if len(w.TPS) >= 1 and w.TPS[0] >= 0:         #如果后面还有已经安排好的任务，则执行后面的任务
                    s = self.S[w.TPS[0]]
                    if t + self.distance(w, s) / w.v < s.t + s.d:
                        w.idle_time = t + self.distance(w, s) / w.v
                        s.state = 1
                        w.dx = (s.x-w.x)*w.v/self.distance(w, s)
                        w.dy = (s.y-w.y)*w.v/self.distance(w, s)
                    else:
                        w.state = 0
                        w.idle_time = t
                else:
                    w.idle_time = t
                    w.state = 0
            elif w.idle_time > t and len(w.TPS) >= 1 and w.state == 1:  #正在完成某个任务
                w.x = self.S[w.TPS[0]].x - (w.idle_time - t) * w.dx
                w.y = self.S[w.TPS[0]].y - (w.idle_time - t) * w.dy
                
            elif w.idle_time <= t and len(w.TPS) == 0:  #工人没有任务，且处于空闲状态
                w.idle_time = t
                w.state = 0
            elif w.idle_time < t and len(w.TPS) >=1 and w.TPS[0] >= 0 and w.state == 0:#工人开始新任务
                s = self.S[w.TPS[0]]
                if t + self.distance(w,s) / w.v < s.t + s.d:
                    s.state = 1
                    w.state = 1
                    w.idle_time = t + self.distance(w, s) / w.v
                    w.dx = (s.x-w.x)*w.v/self.distance(w, s)
                    w.dy = (s.y-w.y)*w.v/self.distance(w, s)
            
            w.RS = []
            if w.t + w.d < t:
                self.Widx.remove(wi)
            if w.state == 1:
                self.next_time = min(self.next_time, w.idle_time)
        
        for si in copy.deepcopy(self.Sidx):
            if si < 0 and del_pred:
                self.Sidx.remove(si)
            if si >=0 :
                s = self.S[si]
                if s.t + s.d < t:
                    self.Sidx.remove(si)   #去除超时的任务

        self.num_w = len(self.Widx)
        self.num_s = len(self.Sidx)
        self.MPVST = [[] for i in range(self.all_num_w)]
        
        self.e = None
        self.ne = None
        self.h = None
        self.idx = 0
        
        self.root_node = []
        self.tree = None
        self.tree_children = None
        self.idxtree = 0
        self.up_num = []   #树上每个节点的最大完成任务数量
        
        self.memorylen = 0
    
    def init_SP(self, t):
        pass
    
    def pred_match(self, task, id, t):
        xi, yi = task[1], task[2]
        best_id = -1
        for i, p in enumerate(self.SP):
            dis = math.sqrt(pow(xi - p.x, 2) + pow(yi-p.y, 2))
            if dis < 0.15 and abs(p.t - task[0]) < 5 and p.t > task[0] and p.w != -1 and p.w in self.Widx:  #刚出现的任务匹配上了之前预测的任务。
                for j, si in enumerate(self.W[p.w].TPS):
                    if - 1 - si == i:
                        if j == 0:
                            print("www")
                        cost_time = 0
                        self.W[p.w].TPS[j] = id   #直接进行分配
                        self.SP[i].w = -1
                        self.acc += 1
                        return False
        return True
        
    def fresh_SP(self, task_data, t):
        with torch.no_grad():
            task_seq = task_test_process(task_data, t, max_num_task=self.arg["max_num_task"], model=self.arg["task_model"], time_interval=self.arg["time_interval"], seq=self.arg['seq'], device=self.arg["device"])
            if task_seq is None:
                return
            out = self.Pmodel(task_seq)
            self.SPgrid = grid_predict_to_task(out, now=t, max_num_task=self.arg["max_num_task"], model=self.arg["task_model"], time_interval=self.arg["time_interval"], threshold=self.arg['threshold'])
            self.sp_idx = 0
            self.SP = []
            for i in range(self.arg['gx']):
                for j in range(self.arg['gy']):
                    for p in self.SPgrid[i][j]:
                        self.SP.append(Task(id = -1 - self.sp_idx, task=[p[1], i * 0.5 + 0.25, j * 0.5 + 0.25, 40, 10], predicted=p[0], deadline=self.arg["deadline"]))
                        self.Sidx.append(-1 - self.sp_idx)
                        self.sp_idx += 1
                        self.num_s += 1
                         
    def get_vis_data(self, t):
        vis_data = []
        for si in self.Sidx:
            s = self.get_s(si)
            if si < 0:
                vis_data.append([s.x, s.y, 2])
            else:
                vis_data.append([s.x, s.y, 0])
        for wi in self.Widx:
            w = self.W[wi]
            vis_data.append([w.x, w.y, 1])
        return vis_data
        
    def add_new(self, newworks, newtasks, t):
        is_task_ass = False
        for w in newworks:
            self.W.append(Worker(self.all_num_w, w, self.arg["reach_distance"], self.arg["wd"]))
            self.Widx.append(self.all_num_w)
            self.all_num_w += 1
            self.num_w += 1
            is_task_ass = True
        for s in newtasks:
            self.S.append(Task(self.all_num_s, s, self.arg["deadline"]))
            self.Sidx.append(self.all_num_s)
            # if self.is_predict:
            #     is_task_ass = self.pred_match(s, id=self.all_num_s, t=t)
            # else:
            is_task_ass = True
            self.all_num_s += 1
            self.num_s += 1
        return is_task_ass
            
    def distance(self, w, s):
        return math.sqrt(pow(w.x-s.x, 2) + pow(w.y-s.y, 2)) + 1e-8
    
    def get_s(self, id):
        if id < 0:
            return self.SP[- 1 - id]
        else:
            return self.S[id]
        
    def get_RS(self, num=5):
        for wi in self.Widx:
            w = self.W[wi]
            w.RS = []
            temp = []
            for si in self.Sidx:
                s = self.get_s(si)
                if s.state == 1:
                    continue
                if len(w.TPS) >=1 and w.state:
                    if self.distance(self.S[w.TPS[0]], s) < w.r and s.t-self.t < 50:
                        temp.append([si, self.distance(self.S[w.TPS[0]], s)])
                elif self.distance(w, s) < w.r and s.t-self.t < 50:
                    temp.append([si, self.distance(w, s)])
            temp.sort(key= lambda x: x[1])
            s_temp = [ss for ss in temp if ss[0] >= 0]
            if len(s_temp) > num:
                s_temp = s_temp[:num]
            sp_temp = [ss for ss in temp if ss[0] < 0]
            if len(sp_temp) > 1:
                sp_temp = sp_temp[:1]
            w.RS = [ss[0]  for ss in s_temp] + [ss[0]  for ss in sp_temp]
    
    def find_all_maximaxl_PVTS(self, w:Worker, lasts, last_t):
        final_PVTS = []
        # all_pred = all([si < 0 for si in w.RS])
        # pred_num = len([x for x in lasts if x < 0])
        
        if len(lasts) >= self.arg["maxPVTS"]:
            return [[lasts, last_t], ]
        
        for si in w.RS:
            if  si in lasts:
                continue
            cost = 0
            s = self.get_s(si)
            if len(lasts) == 0:
                cost = self.distance(w, s) / w.v
            else:
                cost = self.distance(self.get_s(lasts[-1]), s) / w.v
            if si < 0 and  max(w.idle_time, self.t) + last_t < s.t:
                cost += s.t - (max(w.idle_time, self.t) + last_t)
            
            next_idle_time = max(w.idle_time, self.t) + last_t + cost
            if next_idle_time >= w.t + w.d or next_idle_time >= s.t + s.d or next_idle_time >= self.t + self.max_time or cost * w.v > w.r:
                continue
            final_PVTS += self.find_all_maximaxl_PVTS(w, lasts + [si,], last_t + cost)
        
        if len(final_PVTS) == 0:
            return [[lasts, last_t],]
        return final_PVTS
    
    def add(self, a, b):
        self.e[self.idx] = b
        self.ne[self.idx] = self.h[a]
        self.h[a] = self.idx
        self.idx += 1
    
    def get_WDG(self,):
        self.e = np.zeros((self.num_w * self.num_w + 10), dtype=int)
        self.ne = np.zeros((self.num_w * self.num_w + 10), dtype=int)
        self.h = np.full((self.num_w * self.num_w + 10), -1)
        self.idx = 0
        self.lr = []
        for i in range(self.num_w):
            wi = self.Widx[i]
            if len(self.W[wi].RS) == 0:
                continue
            li, ri = min(self.W[wi].RS), max(self.W[wi].RS)
            for j in range(i+1, self.num_w):
                wj = self.Widx[j]
                if len(self.W[wj].RS) == 0:
                    continue
                lj, rj = min(self.W[wj].RS), max(self.W[wj].RS)
                if lj > ri or rj < li:
                    continue
                self.add(i, j)
                self.add(j, i)
    
    def get_WDG2(self,):
        self.e = np.zeros((self.num_w * self.num_w + 10), dtype=int)
        self.ne = np.zeros((self.num_w * self.num_w + 10), dtype=int)
        self.h = np.full((self.num_w * self.num_w + 10), -1)
        self.idx = 0
        self.lr = []
        for i in range(self.num_w):
            wi = self.Widx[i]
            if self.arg["fpta"] and self.W[wi].state == 1:
                    continue
            for j in range(i+1, self.num_w):
                wj = self.Widx[j]
                if len(self.W[wj].RS) == 0 or len(self.W[wi].RS) == 0:
                    continue
                if self.arg["fpta"] and self.W[wj].state == 1:
                    continue
                interset = set(self.W[wi].RS).intersection(set(self.W[wj].RS))
                if len(interset) == 0:
                    continue
                self.add(i, j)
                self.add(j, i)
                            
    def MCS(self, wset):
        v = [[]for i in range(self.num_w)]
        label = [0 for i in range(self.num_w)]
        st = [False for i in range(self.num_w)]
        ans = []
        
        num_w = 0
        for i in range(self.num_w):
            if wset[i] == 1:
                v[0].append(i)
                num_w += 1
        top = 0
        for k in range(num_w, 0, -1):
            now = -1
            while(now == -1):
                for j in reversed(v[top]):
                    if not st[j] and wset[j]:
                        now = j
                        break
                    else:
                        v[top].pop(-1)
                if now == -1:
                    top -= 1
                else:
                    break
            v[top].pop(-1)
            ans.append(now)
            st[now] = True
            i = self.h[now]
            while(i!=-1):
                j = self.e[i]
                if wset[j] and not st[j]:
                    label[j] += 1
                    v[label[j]].append(j)
                    top = max(top, label[j])
                i = self.ne[i]
        
        return list(reversed(ans))
                    
    def num_graph(self, wset):
        graph_set = []
        st = copy.deepcopy(wset)   #0 表示已经访问
        for i in range(len(wset)):
            if wset[i] and st[i]:
                q = [i, ]
                temp_q = []
                while(len(q)):
                    t = q[-1]
                    temp_q.append(t)
                    q.pop()
                    st[t] = 0
                    k = self.h[t]
                    while(k!=-1):
                        j = self.e[k]
                        if wset[j] and st[j]:
                            q.append(j)
                        k = self.ne[k]
                graph_set.append(temp_q)
        return graph_set
    
    def get_mutil_graph(self,):  #将原图分为多个连通图
        G = []
        st = [False for i in range(self.num_w)]
        for i in range(self.num_w):
            if st[i]:
                continue
            q = [i, ]
            graph = [0 for _ in range(self.num_w)]
            while len(q):
                t = q.pop()
                graph[t] = 1
                k = self.h[t]
                st[t] = True
                if sum(graph) >= 12:
                    break
                while k != -1:
                    j = self.e[k]
                    if st[j] == False:
                        q.append(j)
                    k = self.ne[k]
            G.append(graph)
        return G
    
    def get_tree(self, wset):
        if sum(wset) == 0:
            return
        node = self.idxtree
        self.idxtree += 1
        
        ms = self.MCS(wset)  #获取完美消除序列
        #获取达到最平衡的极大团
        
        max_balance_set = []
        max_num = 0
        max_v = 0
        
        for v in ms:
            newset = copy.deepcopy(wset)
            newset[v] = 0
            i = self.h[v]
            while(i!=-1):
                j = self.e[i]
                newset[j] = 0
                i = self.ne[i]
            graph_set = self.num_graph(newset)
            if len(graph_set) + 1 > max_num:
                max_balance_set = graph_set
                max_num = len(graph_set) + 1
                max_v = v
        
        wset[max_v] = 0
        self.tree[node].append(max_v)
        i = self.h[max_v]
        while(i!=-1):
            j = self.e[i]
            if wset[j]:
                wset[j] = 0
                self.tree[node].append(j)
            i = self.ne[i]
        
        for sub_graph in max_balance_set:
            new_sub_set = [1 if i in sub_graph else 0 for i in range(self.num_w)]
            self.tree_children[node].append(self.idxtree)
            self.get_tree(new_sub_set)
            
    def get_up_num(self,node):
        self.up_num[node] = 0
        for i in self.tree[node]:
            if len(self.MPVST[self.Widx[i]]):
                self.up_num[node] += max([len(k[0]) for k in self.MPVST[self.Widx[i]]])
            self.tree_all_children_node[node].append(i)
        for j in self.tree_children[node]:
            up_num, children_node = self.get_up_num(j)
            self.up_num[node] += up_num
            self.tree_all_children_node[node] += children_node
        return self.up_num[node], self.tree_all_children_node[node]
    
    def update_MPTS(self, S, MPVST):
        ans = []
        for q in MPVST:
            for i, si in enumerate(q[0]):
                if si not in S:
                    ans.append([q[0][:i], q[1]])
                    break
                if i==len(q[0])-1:
                    ans.append(q)
        return ans
    
    def dfs(self, node, Sidx, W, h):
        opt = 0
        allocation = {}
        if self.up_num[node] < h:
            return 0, allocation
    
        if len(W) != 0:
            gk = 0
            for i, w in enumerate(W):
                if gk > 1:
                    break
                if len(W) > 1 and gk > 0:
                    break
                if w.state==1 and self.arg["fpta"]:
                    continue
                gk += 1
                tempW = W[:]
                tempW.pop(i)
                Q = self.update_MPTS(Sidx, self.MPVST[w.id])
                if len(Q) == 0:
                    return self.dfs(node, copy.deepcopy(Sidx), tempW, h)
                for qc in Q:
                    q = qc[0]
                    dopt = len(q)
                    new_opt, new_allocation =  self.dfs(node, list(set(Sidx).difference(q)),  tempW, h-dopt)
        
                    if new_opt + dopt > opt:
                        allocation = new_allocation
                        allocation[str(w.id)] = q
                        opt = new_opt + dopt
                    h = max(h, opt)
                    if self.is_train_RL and self.memorylen < self.max_mem_len:
                        #获取价值函数训练数据
                        task_state = []
                        for j in Sidx:
                            s = self.get_s(j)
                            task_state.append([(s.t-self.t)/self.max_time, s.x/self.max_x, s.y/self.max_y, s.d/self.max_time])
                            if len(task_state) > self.arg['nt']:
                                break
                        worker_state = []
                        for j in range(0, len(W)):
                            ww = W[j]
                            worker_state.append([(ww.idle_time-self.t)/self.max_time, ww.x/self.max_x, ww.y/self.max_y, ww.r/self.max_x, ww.d/self.max_time, ww.v/self.max_x])
                            if len(worker_state) >= self.arg['nw']:
                                break
                        for child in self.tree_children[node]:
                            for j in self.tree_all_children_node[child]:
                                ww = self.W[self.Widx[j]]
                                worker_state.append([(ww.idle_time-self.t)/self.max_time, ww.x/self.max_x, ww.y/self.max_y, ww.r/self.max_x, ww.d/self.max_time, ww.v/self.max_x])
                                if len(worker_state) >= self.arg['nw']:
                                    break
        
                        action_q = []
                        for j in q:
                            s = self.get_s(j)
                            action_q.append([(s.t-self.t)/self.max_time, s.x/self.max_x, s.y/self.max_y, s.d/self.max_time])
                            if len(action_q) > self.arg['nq']:
                                break
                        action_w = [(w.idle_time-self.t)/self.max_time, w.x/self.max_x, w.y/self.max_y, w.r/self.max_x, w.d/self.max_time, w.v/self.max_x]
                        self.memorylen += 1
                        self.memory.append([task_state, worker_state, action_w, action_q, new_opt + len(q)])
        else:
            for i, child in enumerate(self.tree_children[node]):
                LB = h - opt
                for j in range(i+1, len(self.tree_children[node])):
                    LB -= self.up_num[self.tree_children[node][j]]
                WN = [self.W[self.Widx[id]] for id in self.tree[child]]
                new_opt, new_allocation = self.dfs(child, Sidx, WN, LB)
                opt += new_opt
                for k, wi in enumerate(self.Widx):
                    if k in self.tree_all_children_node[child] and new_allocation.get(str(wi)):
                        allocation[str(wi)] = new_allocation[str(wi)]
        return opt, allocation
            
    def bfs(self, node, Sidx, W):
        opt = 0
        allocation = {}
        if len(W) != 0:
            w = W[0]
            Q = self.update_MPTS(Sidx, self.MPVST[w.id])
            if len(Q) == 0:
                return self.bfs(node, copy.deepcopy(Sidx), [tw for tw in W if tw != w])
            
            task_state = []
            for j in Sidx:
                s = self.get_s(j)
                task_state.append([(s.t-self.t)/self.max_time, s.x/self.max_x, s.y/self.max_y, s.d/self.max_time])
            worker_state = []
            for j in range(0, len(W)):
                ww = W[j]
                worker_state.append([(ww.idle_time-self.t)/self.max_time, ww.x/self.max_x, ww.y/self.max_y, ww.r/self.max_x, ww.d/self.max_time, ww.v/self.max_x])
            for child in self.tree_children[node]:
                for j in self.tree_all_children_node[child]:
                    ww = self.W[self.Widx[j]]
                    worker_state.append([(ww.idle_time-self.t)/self.max_time, ww.x/self.max_x, ww.y/self.max_y, ww.r/self.max_x, ww.d/self.max_time, ww.v/self.max_x])
            action_w = [(w.idle_time-self.t)/self.max_time, w.x/self.max_x, w.y/self.max_y, w.r/self.max_x, w.d/self.max_time, w.v/self.max_x]
            best_q = Q[0][0]
            for qc in Q:
                q = qc[0]
                action_q = []
                for j in q:
                    s = self.get_s(j)
                    action_q.append([(s.t-self.t)/self.max_time, s.x/self.max_x, s.y/self.max_y, s.d/self.max_time])
                
                temp_data = [[task_state, worker_state, action_w, action_q], ]
                Qdata = qdata_process(temp_data, nt=self.arg["nt"], nw=self.arg["nw"], nq=self.arg["nq"], is_train=False, device=self.arg["device"])
                Qvalue = self.Qmodel(Qdata[0], Qdata[1], Qdata[2], Qdata[3])
                Qvalue = Qvalue[0].item()
                
                if Qvalue > opt:
                    best_q = q
                    opt = Qvalue
            new_allocation = self.bfs(node, list(set(Sidx).difference(best_q)), [tw for tw in W if tw != w])
            allocation = new_allocation
            allocation[str(w.id)] = best_q
        else:            
            for i, child in enumerate(self.tree_children[node]):
                WN = [self.W[self.Widx[id]] for id in self.tree[child]]
                new_allocation = self.bfs(child, Sidx, WN)
                for k, wi in enumerate(self.Widx):
                    if k in self.tree_all_children_node[child] and new_allocation.get(str(wi)):
                        allocation[str(wi)] = new_allocation[str(wi)]
        return allocation
    
    def greedy(self, Sidx):
        opt = 0
        allocation = {}
        for wi in self.Widx:
            w = self.W[wi]
            Q = self.update_MPTS(Sidx, self.MPVST[w.id])
            Q.sort(key=lambda x: len(x[0]))
            Q.reverse()
            if len(Q) == 0 or w.state == 1:
                continue
            q = Q[0][0]
            opt += len(q)
            allocation[str(w.id)] = q
            Sidx = list(set(Sidx).difference(q))
        return opt, allocation
        
    def best_vsts(self, w, begin, vst, last_t, first=False):
        out = []
        if len(vst) == 0:
            return vst, 0
        cost = 1000
        for i, v in enumerate(vst):
            # if first and self.get_s(vst[i]).id < 0:
            #     continue
            s = self.get_s(vst[i])
            cost2 = 0
            cost2 = self.distance(begin, self.get_s(v))/w.v
            if s.id < 0 and max(w.idle_time, self.t) + last_t < s.t:
                cost2 += s.t - (max(w.idle_time, self.t) + last_t)
            
            temp = copy.copy(vst)
            temp.pop(i)
            v_temp, dis = self.best_vsts(w, s, temp, last_t + cost2)
            
            if dis + cost2 < cost:
                cost = dis + cost2
                out = [vst[i]] + v_temp
        return out, dis + cost2
    
    def assign(self, t):
        self.t = t
        if len(self.S) == 0:
            return 0
        #获得每个工人的RS
        self.get_RS(num=self.arg["maxRS"])

        #获取每个工人的Maximal_P-VTS
        time1 = time.time()
        
        for wi in self.Widx:
            w = self.W[wi]
            if w.state and len(w.TPS):
                temp =  self.find_all_maximaxl_PVTS(w, [w.TPS[0],], 0)
            else:
                temp =  self.find_all_maximaxl_PVTS(w, [], 0)
            # print(len(w.RS))
            # print(len(temp))
            
            route = list(set(tuple(sorted(item[0])) for item in temp))  #去除相同任务集合
            tm = time.time()
            
            temp2 = []
            for vst in route:       #去除任务集合相同，但是花费多的任务序列
                vst = list(vst)
                if len(vst) == 0:
                    continue
                if w.state and len(w.TPS):
                    vst.remove(w.TPS[0])
                    if len(vst):
                        vst_temp, cost = self.best_vsts(w, self.get_s(w.TPS[0]), vst, 0, True,)
                        temp2.append([[w.TPS[0],] + vst_temp, cost])
                else:
                    if len(vst):
                        vst_temp, cost = self.best_vsts(w, w, vst, 0, True)
                        temp2.append([vst_temp, cost])
            if w.state and len(w.TPS):
                temp2.append([[w.TPS[0],], 0])     
            # ttt += time.time() -tm
            temp2.sort(key=lambda x: x[1])
            self.MPVST[wi] = temp2[:10]
            self.ttt = max(self.ttt, len(temp2))
        if t >= 3000:
            time2 = time.time()
        #构建依赖图
        self.get_WDG2()
        
        time3 = time.time()
        #树分解
        self.tree = [[] for i in range(self.num_w)]
        self.tree_children = [[] for i in range(self.num_w)]
        self.tree_all_children_node = [[] for i in range(self.num_w)]
        self.up_num = [0 for i in range(self.num_w)]
        self.idxtree = 0
        
        G = self.get_mutil_graph()
        if t == 450:
            time4 = time.time()
        temp_sidx = copy.deepcopy(self.Sidx)
        
        if self.is_greedy:
            newopt, allocation =  self.greedy(temp_sidx)
            self.Widx.sort(key=lambda x: len(self.MPVST[x]))
            for i, wi in enumerate(self.Widx):

                if allocation.get(str(wi)):
                    self.W[wi].TPS = allocation[str(wi)]
                elif self.W[wi].state == 0:
                    self.W[wi].TPS = []

                for si in self.W[wi].TPS:
                    if si < 0:
                        self.SP[- 1 - si].w = wi
                    else:
                        self.get_s(si).state = 1
                    if si in temp_sidx:
                        temp_sidx.remove(si)
        else:
            for g in G:
                self.root_node.append(self.idxtree)
                self.get_tree(copy.deepcopy(g))
                # if self.t>500:
                #     plot_tree(self.tree_children,self.root_node[-1])
                self.get_up_num(self.root_node[-1])
                #DFS搜索最优结果
                WN = [self.W[wi] for i, wi in enumerate(self.Widx) if g[i] and i in self.tree[self.root_node[-1]]]
                WN.sort(key = lambda x : len(self.MPVST[x.id]))
                WN.reverse()
                if self.is_bfs:
                    allocation =  self.bfs(self.root_node[-1], temp_sidx, WN)
                else:
                    newopt, allocation =  self.dfs(self.root_node[-1], temp_sidx, WN, 0)
                for i, wi in enumerate(self.Widx):
                    if g[i]:
                        if self.W[wi].state and allocation.get(str(wi))==None and not self.arg["fpta"]: 
                            print('error')
                        
                        if self.arg["fpta"]:
                            if allocation.get(str(wi)):
                                self.W[wi].TPS = allocation[str(wi)]
                            elif self.W[wi].state == 0:
                                self.W[wi].TPS = []
                        else:
                            if allocation.get(str(wi)):
                                self.W[wi].TPS = allocation[str(wi)]
                            else:
                                self.W[wi].TPS = []
                            
                        for si in self.W[wi].TPS:
                            if si < 0:
                                self.SP[- 1 - si].w = wi
                            elif self.arg["fpta"]:
                                self.get_s(si).state = 1
                            if si in temp_sidx:
                                temp_sidx.remove(si)
        
        time5 = time.time()
        #更新每个用户的空闲时刻
        self.next_time = t + 500
        # self.init(t)
    
        print("-"*20)
        print(f"tree_num:{self.idxtree},time:{t},opt={self.R},acc={self.acc}")
        #print(f"tt: {self.ttt}, get_WDG:{time2-time1}, TD cost:{time4-time3},TA cost:{time5 - time4}")

if __name__ == "__main__":
    pass