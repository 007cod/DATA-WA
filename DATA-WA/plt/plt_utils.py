import matplotlib.pyplot as plt

def parse_data(file_path, id):
    data = {
        "Greedy": [],
        "FTA": [],
        "DTA": [],
        "DTA+TP": [],
        "DATA-WA": [],  # Add BFS to the data dictionary
    }
    data_time = {
        "Greedy": [], 
        "FTA": [],
        "DTA": [],
        "DTA+TP": [],
        "DATA-WA": [],  # Add BFS to the data_time dictionary
    }
    data_num_set = set()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(',')
            try:
                data_num = int(parts[id].split(':')[1].strip())
            except:
                data_num = float(parts[id].split(':')[1].strip())
            
            R = int(parts[2].split(':')[1].strip())
            avgTime = float(parts[4].split(':')[1].strip())
            GTA = parts[5].split(':')[1].strip() == 'True'
            FTA = parts[6].split(':')[1].strip() == 'True'
            predict = parts[7].split(':')[1].strip() == 'True'
            bfs = parts[8].split(':')[1].strip() == 'True'
            
            data_num_set.add(data_num)
            
            if GTA:
                data["Greedy"].append((data_num, R))
                data_time["Greedy"].append((data_num, avgTime))
            elif FTA:
                data["FTA"].append((data_num, R))
                data_time["FTA"].append((data_num, avgTime))
            elif bfs:
                data["DATA-WA"].append((data_num, R))
                data_time["DATA-WA"].append((data_num, avgTime))
            elif not predict:
                data["DTA"].append((data_num, R))
                data_time["DTA"].append((data_num, avgTime))
            else:
                data["DTA+TP"].append((data_num, R))
                data_time["DTA+TP"].append((data_num, avgTime))

    return data, data_time, sorted(data_num_set)


# Plotting function
def plot_data(data, data_nums, temp_data_nums, data_name, datatype, miny, maxy, x_lable):
    plt.figure(figsize=(5, 5))

    # Separate data by data_num for plotting
    Greedy = [r for dn, r in sorted(data["Greedy"])]
    FTA = [r for dn, r in sorted(data["FTA"])]
    DTA = [r for dn, r in sorted(data["DTA"])]
    DTA_TP = [r for dn, r in sorted(data["DTA+TP"])]
    BFS = [r for dn, r in sorted(data["DATA-WA"])]
    # Plotting each line
    plt.plot(temp_data_nums, Greedy, '^-', label='Greedy', color='purple', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, FTA, 'o-', label='FTA', color='green', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, DTA, 's-', label='DTA', color='blue', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, DTA_TP, 'd-', label='DTA+TP', color='red', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, BFS, 'x-', label='DATA-WA', color='orange', markersize=8, markerfacecolor='none')

    # Adding labels and title
    plt.xlabel(x_lable, fontsize=12)
    plt.ylabel('Number of Assigned Tasks', fontsize=12)
    plt.xticks(temp_data_nums, data_nums)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Setting the y-axis range
    plt.ylim(miny, maxy)  # Customize the range as needed

    # Adding a legend outside the plot at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.15), ncol=3)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'./plt/{data_name}/output/{datatype}_{data_name}_R.pdf')
    
    plt.show()
    
    
# Plotting function
def plot_time(data, data_nums, temp_data_nums, data_name, datatype, miny, maxy, x_lable):
    plt.figure(figsize=(5, 5))

    # Separate data by data_num for plotting
    Greedy = [r for dn, r in sorted(data["Greedy"])]
    FTA = [r for dn, r in sorted(data["FTA"])]
    DTA = [r for dn, r in sorted(data["DTA"])]
    DTA_TP = [r for dn, r in sorted(data["DTA+TP"])]
    BFS = [r for dn, r in sorted(data["DATA-WA"])]

    # Plotting each line
    plt.plot(temp_data_nums, Greedy, '^-', label='Greedy', color='purple', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, FTA, 'o-', label='FTA', color='green', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, DTA, 's-', label='DTA', color='blue', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, DTA_TP, 'd-', label='DTA+TP', color='red', markersize=8, markerfacecolor='none')
    plt.plot(temp_data_nums, BFS, 'x-', label='DATA-WA', color='orange', markersize=8, markerfacecolor='none')

    # Adding labels and title
    plt.xlabel(x_lable, fontsize=12)
    plt.ylabel('CPU Time (s)', fontsize=12)
    plt.xticks(temp_data_nums, data_nums)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Setting the y-axis range
    plt.ylim(miny, maxy)  # Customize the range as needed

    # Adding a legend outside the plot at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(f'./plt/{data_name}/output/{datatype}_{data_name}_cpu.pdf')
    plt.show()