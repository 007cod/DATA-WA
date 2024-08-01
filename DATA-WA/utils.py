import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np
import torch
import torch.nn as nn

def action_vis(data, reward):
    # 创建一个图形和一个轴
    fig, ax = plt.subplots()

    # 创建初始的散点图
    sc_worker = ax.scatter([], [], marker='o', color='blue')
    sc_task = ax.scatter([], [], marker='*', color='red')
    task_pred = ax.scatter([], [], marker='D', color='green')

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, color='black')
    reward_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=10, color='black')
    # 设置图形边界
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)

    # 更新函数：更新散点的位置
    def update(frame):
        if len(data[frame]):
            x, y, c = zip(*data[frame])
        else:
            x, y, c = [], [], []
        sc_worker.set_offsets(np.column_stack(([d for i, d in enumerate(x) if c[i]==1 ], [d for i, d in enumerate(y) if c[i]==1 ])))
        sc_task.set_offsets(np.column_stack(([d for i, d in enumerate(x) if c[i]==0 ], [d for i, d in enumerate(y) if c[i]==0 ])))
        task_pred.set_offsets(np.column_stack(([d for i, d in enumerate(x) if c[i]==2 ], [d for i, d in enumerate(y) if c[i]==2 ])))
        time_text.set_text(f'Time: {frame}')  # 更新时间标签文本
        reward_text.set_text(f'R: {reward[frame]}') 
        return sc_worker, sc_task, task_pred, time_text, reward_text,

    # 创建动画对象
    ani = FuncAnimation(fig, update, frames=range(len(data)), interval=40, blit=True)

    # 显示动画
    plt.show()
    
def qdata_process(data, nt, nw, nq, is_train=True, device='cpu'):
    if is_train:
        task_state, worker_state, action_w, action_q, r = zip(*data)
    else:
        task_state, worker_state, action_w, action_q = zip(*data)
    b = len(task_state)    
    temp_task_state = torch.zeros(b, nt * 4).to(device)
    temp_worker_state =  torch.zeros(b, nw * 6).to(device)
    temp_action_q = torch.zeros(b, nq * 4).to(device)
    for i in range(b):
        k = min(len(task_state[i])*4, nt * 4)
        if k:
            temp_task_state[i,:k] = torch.tensor(task_state[i]).view(-1)[:k]
        
        k = min(len(worker_state[i])*6, nw * 6)
        if k:
            temp_worker_state[i,:k] = torch.tensor(worker_state[i]).view(-1)[:k]
        
        k = min(len(action_q[i])*4, nq * 4)
        if k:
            temp_action_q[i,:k] = torch.tensor(action_q[i]).view(-1)[:k]

    action_w = torch.tensor(action_w).float().view(b, 6).to(device)
    # action_q =  torch.tensor(action_q[:5]).view(s, -1)
    # next_task_state =  torch.tensor(next_task_state[:nt]).view(s, -1)
    # next_worker_state =  torch.tensor(next_worker_state[:nw]).view(s, -1)
    if is_train:
        r = torch.tensor(r).float().to(device)
        return [temp_task_state, temp_worker_state, action_w, temp_action_q], r
    else:
        return [temp_task_state, temp_worker_state, action_w, temp_action_q]
    


def draw_tree(node, idx, x, y, ax, angle_range=180, radius=1.0):
    # Draw the node
    ax.plot(x, y, 'o')
    
    # Get the number of children
    num_children = len(node[idx])
    
    # If the node has no children, then we're done
    if num_children == 0:
        return
    
    # Compute the angle step
    angle_step = angle_range / max(1, num_children - 1)
    
    # Draw each child
    for i, child in enumerate(node[idx]):
        # Compute the angle
        angle = np.radians(i * angle_step - angle_range / 2)
        
        # Compute the position of the child
        child_x = x + radius * np.cos(angle)
        child_y = y + radius * np.sin(angle)
        
        # Draw a line from the node to the child
        ax.plot([x, child_x], [y, child_y], 'k-')
        
        # Draw the subtree starting at the child
        draw_tree(node, child, child_x, child_y, ax, angle_range=angle_step, radius=radius*0.6)

def plot_tree(tree, root):
    import copy
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Draw the tree
    draw_tree(tree, root, 0, 0, ax)

    # Show the plot
    plt.show()