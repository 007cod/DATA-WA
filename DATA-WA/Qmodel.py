import torch
import torch.nn as nn

class Qnet(nn.Module):
    def __init__(self, nt, nw, nq, hidden_size):
        super(Qnet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(nt*4 , hidden_size),
            nn.ReLU(),
        )
        self.l2 = nn.Sequential(
            nn.Linear(nw*6 , hidden_size),
            nn.ReLU(),
        )
        self.l3 = nn.Sequential(
            nn.Linear(6 , hidden_size),
            nn.ReLU(),
        )
        self.l4 = nn.Sequential(
            nn.Linear(nq*4 , hidden_size),
            nn.ReLU(),
        )
        self.final_l = nn.Sequential(
            nn.Linear(hidden_size*4, hidden_size),
            nn.ReLU(),  
            nn.Linear(hidden_size, 1),
        )
 
    def forward(self, task_state, worker_state, action_w, action_q):
        task_state = self.l1(task_state)
        worker_state = self.l2(worker_state)
        action_w = self.l3(action_w)
        action_q = self.l4(action_q)
        out = torch.concat([task_state, worker_state, action_w, action_q], dim=-1)
        out = self.final_l(out)
        return out