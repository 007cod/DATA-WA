#!/bin/bash
# -*- coding: utf-8 -*-

# 定义固定参数
time_interval=100
seq=5
gx=10
gy=10
max_num_task=20
grid_length=0.5
nw=10
nt=20
nq=4
maxRS=4
maxPVTS=2
threshold=0.85
task_model="GNN"
data_path="./dataset/yc/yc.txt"
model_weight_path="./TaskP/output/GNN4_best_yc2_1002.pth"  #"./TaskP/output/GNN4_best_yc2_1002.pth"
qmodel_path="./RL_model/RL_yc_best_60_52.pth"    # "./RL_model/RL_yc_best_60_52.pth"
device="cpu"
log_path="./plt/yc/wd_yc.txt"
task_num=-1
worker_num=-1
reach_distance=0.51
deadline=40
# 批量执行任务0.51 0.765 1.02 1.265 1.53
for wd in  0.75 1.25 1.5 ; do
  #greedy
  # python main.py \
  #   --time_interval $time_interval \
  #   --seq $seq \
  #   --gx $gx \
  #   --gy $gy \
  #   --max_num_task $max_num_task \
  #   --grid_length $grid_length \
  #   --nw $nw \
  #   --nt $nt \
  #   --nq $nq \
  #   --maxRS $maxRS \
  #   --maxPVTS $maxPVTS \
  #   --is_greedy \
  #   --threshold $threshold \
  #   --task_model $task_model \
  #   --data_path $data_path \
  #   --model_weight_path $model_weight_path \
  #   --qmodel_path $qmodel_path \
  #   --device $device \
  #   --worker_num $worker_num \
  #   --task_num $task_num \
  #   --reach_distance $reach_distance \
  #   --deadline $deadline \
  #   --wd $wd \
  #   --log_path $log_path

  # # 情况1: fpta=true, is_predict=false
  # python main.py \
  #   --time_interval $time_interval \
  #   --seq $seq \
  #   --gx $gx \
  #   --gy $gy \
  #   --max_num_task $max_num_task \
  #   --grid_length $grid_length \
  #   --nw $nw \
  #   --nt $nt \
  #   --nq $nq \
  #   --maxRS $maxRS \
  #   --maxPVTS $maxPVTS \
  #   --fpta \
  #   --threshold $threshold \
  #   --task_model $task_model \
  #   --data_path $data_path \
  #   --model_weight_path $model_weight_path \
  #   --qmodel_path $qmodel_path \
  #   --device $device \
  #   --worker_num $worker_num \
  #   --task_num $task_num \
  #   --reach_distance $reach_distance \
  #   --deadline $deadline \
  #   --wd $wd \
  #   --log_path $log_path

  # #情况2: fpta=false, is_predict=false
  # python main.py \
  #   --time_interval $time_interval \
  #   --seq $seq \
  #   --gx $gx \
  #   --gy $gy \
  #   --max_num_task $max_num_task \
  #   --grid_length $grid_length \
  #   --nw $nw \
  #   --nt $nt \
  #   --nq $nq \
  #   --maxRS $maxRS \
  #   --maxPVTS $maxPVTS \
  #   --threshold $threshold \
  #   --task_model $task_model \
  #   --data_path $data_path \
  #   --model_weight_path $model_weight_path \
  #   --qmodel_path $qmodel_path \
  #   --device $device \
  #   --worker_num $worker_num \
  #   --task_num $task_num \
  #   --reach_distance $reach_distance \
  #   --deadline $deadline \
  #   --wd $wd \
  #   --log_path $log_path

  #情况3: fpta=false, is_predict=true
  python main.py \
    --time_interval $time_interval \
    --seq $seq \
    --gx $gx \
    --gy $gy \
    --max_num_task $max_num_task \
    --grid_length $grid_length \
    --nw $nw \
    --nt $nt \
    --nq $nq \
    --maxRS $maxRS \
    --maxPVTS $maxPVTS \
    --is_predict \
    --threshold $threshold \
    --task_model $task_model \
    --data_path $data_path \
    --model_weight_path $model_weight_path \
    --qmodel_path $qmodel_path \
    --device $device \
    --worker_num $worker_num \
    --task_num $task_num \
    --reach_distance $reach_distance \
    --deadline $deadline \
    --wd $wd \
    --log_path $log_path
  
  # bfs
  python main.py \
    --time_interval $time_interval \
    --seq $seq \
    --gx $gx \
    --gy $gy \
    --max_num_task $max_num_task \
    --grid_length $grid_length \
    --nw $nw \
    --nt $nt \
    --nq $nq \
    --maxRS $maxRS \
    --maxPVTS $maxPVTS \
    --is_predict \
    --is_bfs \
    --threshold $threshold \
    --task_model $task_model \
    --data_path $data_path \
    --model_weight_path $model_weight_path \
    --qmodel_path $qmodel_path \
    --device $device \
    --worker_num $worker_num \
    --task_num $task_num \
    --reach_distance $reach_distance \
    --deadline $deadline \
    --wd $wd \
    --log_path $log_path
done
