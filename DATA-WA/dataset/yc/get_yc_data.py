import os 
import random
import numpy as np

s_path = "./YCN01_S.txt"
w_path = "./YCN01_W.txt"

tasks = []
workers = []
write_items = []
gf = open("./yc_train.txt", 'w')

with open(s_path, 'r') as f:
    for data in f.readlines():
        data = data.strip("\n")
        data = data.split(" ")
        t, x, y, v = int(data[2]), float(data[4]), float(data[5]), float(data[-1])
        tasks.append([t,x,y,v])
    
with open(w_path, 'r') as f:
    for data in f.readlines():
        data = data.strip("\n")
        data = data.split(" ")
        t, x, y, r = int(data[3]), float(data[4]), float(data[5]), float(data[2])
        workers.append([t,x,y,r,0])

tasks = [x  for x in tasks if x[0] > 1477990800-3600 and x[0] < 1477990800 ]
workers = [x  for x in workers if x[0] > 1477990800-3600 and x[0] < 1477990800 ]

items = workers + tasks

items.sort(key= lambda x :x[0])


mint = min([it[0] for it in items])
minx = min([it[1] for it in items])
miny = min([it[2] for it in items])
maxx = max([it[1] for it in items])
maxy = max([it[2] for it in items])

fact = 5 / max(maxy - miny, maxx - minx) - 0.001
print(f"fact: {fact}")
print(f"max x:{maxx}, max y:{maxy}, min x:{minx}, min y:{miny}")

write_items.append(f"task num:{len(tasks)}, worker num:{len(workers)}, fact: {fact}\n")

for item in items:
    item[0] -= mint
    item[1] -= minx
    item[2] -= miny
    if (len(item)==5):
        write_items.append("{} w {:2f} {:2f} 1 1 300 {:2f}\n".format(item[0], item[1]*fact, item[2]*fact, random.random()))
    else:
        write_items.append("{} t {:2f} {:2f} 300 {:2f}\n".format(item[0], item[1]*fact, item[2]*fact, random.random()))


gf.writelines(write_items)  