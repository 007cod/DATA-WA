import matplotlib.pyplot as plt
from plt_utils import parse_data, plot_data, plot_time

if __name__ == '__main__':

    data, data_time, data_nums = parse_data('./plt/yc/w_yc.txt', 1)
    plot_data(data, data_nums, data_nums, "yc", "w", 4800, 7800, "Number of workers")
    plot_time(data_time, data_nums, data_nums, "yc", "w", 0, 1, "Number of workers")
    
    data, data_time, data_nums = parse_data('./plt/yc/t.txt', 0)
    plot_data(data, data_nums, data_nums, "yc", "t", 4300, 7800, "Number of tasks")
    plot_time(data_time, data_nums, data_nums, "yc", "t", 0, 1, "Number of tasks")
    
    data, data_time, data_nums = parse_data('./plt/yc/wd_yc.txt', 9)
    plot_data(data, data_nums, data_nums, "yc", "wd", 4800, 7800, "Available time of workers (h)")
    plot_time(data_time, data_nums, data_nums, "yc", "wd", 0, 1, "Available time of workers (h)")
    
    data, data_time, data_nums = parse_data('./plt/yc/dis_yc.txt', 0)
    plot_data(data, data_nums, list(range(len(data_nums))), "yc", "dis", 3000, 8000, "Reachable distance of workers")
    plot_time(data_time, data_nums, list(range(len(data_nums))), "yc", "dis", 0, 1, "Reachable distance of workers")
    
    data, data_time, data_nums = parse_data('./plt/yc/dead_yc.txt', 1)
    plot_data(data, data_nums, data_nums, "yc", "dead", 3300, 8300, 'Vaild time of task')
    plot_time(data_time, data_nums, data_nums, "yc", "dead", 0, 1, 'Vaild time of task')
    
    
    
    data, data_time, data_nums = parse_data('./plt/dd/w_dd.txt', 1)
    plot_data(data, data_nums, data_nums, "dd", "w", 4800, 7100, "Number of workers")
    plot_time(data_time, data_nums, data_nums, "dd", "w", 0, 1, "Number of workers")
    
    data, data_time, data_nums = parse_data('./plt/dd/t.txt', 0)
    plot_data(data, data_nums, data_nums, "dd", "t", 3500, 7200, "Number of tasks")
    plot_time(data_time, data_nums, data_nums, "dd", "t", 0, 1, "Number of tasks")
    
    data, data_time, data_nums = parse_data('./plt/dd/wd_dd.txt', 9)
    plot_data(data, data_nums, data_nums, "dd", "wd", 4800, 7200, "Available time of workers (h)")
    plot_time(data_time, data_nums, data_nums, "dd", "wd", 0, 1, "Available time of workers (h)")
    
    data, data_time, data_nums = parse_data('./plt/dd/dis_dd.txt', 0)
    plot_data(data, data_nums,  list(range(len(data_nums))), "dd", "dis", 3000, 7200, "Reachable distance of workers")
    plot_time(data_time, data_nums,  list(range(len(data_nums))), "dd", "dis", 0, 1, "Reachable distance of workers")
    
    data, data_time, data_nums = parse_data('./plt/dd/dead_dd.txt', 1)
    plot_data(data, data_nums,  data_nums, "dd", "dead", 3300, 7800, "Vaild time of task")
    plot_time(data_time, data_nums,  data_nums, "dd", "dead", 0, 1, "Vaild time of task")