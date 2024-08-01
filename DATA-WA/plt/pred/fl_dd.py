import matplotlib.pyplot as plt

    # data = {
    #     "LSTM_dd": [0.43, 0.52, 0.62, 0.68, 0.7, 0.77],
    #     "WaveNet_dd": [0.46, 0.51, 0.60, 0.65, 0.75, 0.80],
    #     "DDGNN_dd": [0.62, 0.69, 0.75, 0.81, 0.85, 0.88],
    #     "LSTM_yc": [0.45, 0.53, 0.62, 0.69, 0.74, 0.8 ],
    #     "WaveNet_yc": [0.47, 0.50, 0.61, 0.67, 0.75, 0.82],
    #     "DDGNN_yc": [0.59,0.67,0.76,0.81,0.84,0.88],
    # }
# Function to parse the data from the provided text
def parse_data():
    data = {
        "LSTM_dd": [0.43, 0.49, 0.60, 0.63, 0.7],
        "WaveNet_dd": [0.46, 0.51, 0.61, 0.65, 0.75],
        "DDGNN_dd": [0.62, 0.69, 0.75, 0.81, 0.85],
    }
    
    assigned_data = {
        "LSTM_dd": [6788, 6762, 6800, 6785, 6766],
        "WaveNet_dd": [6805, 6814, 6822, 6786, 6779],
        "DDGNN_dd": [6970, 6901, 6945, 6868, 6910],
    }
    
    train_time = {
        "LSTM_dd": [0.046, 0.040, 0.034, 0.031, 0.026],
        "WaveNet_dd": [18.48, 15.68, 14.43, 10.83, 10.8],
        "DDGNN_dd": [21.97, 19.12, 17.94, 15.42, 13.28],
    }
    
    test_time = {
        "LSTM_dd": [0.001, 0.001, 0.001, 0.001, 0.001],
        "WaveNet_dd": [0.073, 0.075, 0.071, 0.071, 0.071],
        "DDGNN_dd": [0.073, 0.071, 0.0724, 0.072, 0.074],
    }
        
    data_num_set = [5, 6, 7, 8, 9]

    return data_num_set, data, assigned_data, train_time, test_time

# Plotting function
def plot_data(data, data_nums):
    plt.figure(figsize=(6, 5))

    # Separate data by data_num for plotting
    LSTM_dd = data["LSTM_dd"]
    DDGNN_dd = data["DDGNN_dd"]
    WaveNet_dd = data["WaveNet_dd"]

    # Plotting each line

    plt.plot(data_nums, LSTM_dd, '^-', label='LSTM', color='blue', markersize=8)
    plt.plot(data_nums, WaveNet_dd, 'o-', label='Graph-Wavenet', color='orange', markersize=8)
    plt.plot(data_nums, DDGNN_dd, 'd-', label='DDGNN', color='red', markersize=8)

    # Adding labels and title
    plt.xlabel('Time interval', fontsize=12)
    plt.ylabel('Average Precision', fontsize=12)
    plt.xticks(data_nums)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Setting the y-axis range
    plt.ylim(0.4, 1)  # Customize the range as needed

    # Adding a legend outside the plot at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=3)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('./plt/pred_AP_dd.pdf')
    
    plt.show()
    
# Plotting function
def plot_assign(data, data_nums):
    plt.figure(figsize=(6, 5))

    LSTM_dd = data["LSTM_dd"]
    DDGNN_dd = data["DDGNN_dd"]
    WaveNet_dd = data["WaveNet_dd"]

    # Plotting each line

    plt.plot(data_nums, LSTM_dd, '^-', label='LSTM', color='blue', markersize=8)
    plt.plot(data_nums, WaveNet_dd, 'o-', label='Graph-Wavenet', color='orange', markersize=8)
    plt.plot(data_nums, DDGNN_dd, 'd-', label='DDGNN', color='red', markersize=8)

    # Adding labels and title
    plt.xlabel('Time interval', fontsize=12)
    plt.ylabel('Number of Assigned Tasks', fontsize=12)
    plt.xticks(data_nums)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Setting the y-axis range
    plt.ylim(6600, 7100)  # Customize the range as needed

    # Adding a legend outside the plot at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=3)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('./plt/pred_R_dd.pdf')
    
    plt.show()
    

def plot_train_time(data, data_nums):
    plt.figure(figsize=(6, 5))

    # Separate data by data_num for plotting
    LSTM_dd = data["LSTM_dd"]
    DDGNN_dd = data["DDGNN_dd"]
    WaveNet_dd = data["WaveNet_dd"]

    # Plotting each line

    plt.plot(data_nums, LSTM_dd, '^-', label='LSTM', color='blue', markersize=8)
    plt.plot(data_nums, WaveNet_dd, 'o-', label='Graph-Wavenet', color='orange', markersize=8)
    plt.plot(data_nums, DDGNN_dd, 'd-', label='DDGNN', color='red', markersize=8)
    

    # Adding labels and title
    plt.xlabel('Time interval', fontsize=12)
    plt.ylabel('Training Time (h)', fontsize=12)
    plt.xticks(data_nums)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Setting the y-axis range
    plt.ylim(0, 30)  # Customize the range as needed

    # Adding a legend outside the plot at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=3)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('./plt/pred_train_time_dd.pdf')
    
    plt.show()
    
def plot_test_time(data, data_nums):
    plt.figure(figsize=(6, 5))

    # Separate data by data_num for plotting
    LSTM_dd = data["LSTM_dd"]
    DDGNN_dd = data["DDGNN_dd"]
    WaveNet_dd = data["WaveNet_dd"]

    # Plotting each line

    plt.plot(data_nums, LSTM_dd, '^-', label='LSTM', color='blue', markersize=8)
    plt.plot(data_nums, WaveNet_dd, 'o-', label='Graph-Wavenet', color='orange', markersize=8)
    plt.plot(data_nums, DDGNN_dd, 'd-', label='DDGNN', color='red', markersize=8)
    

    # Adding labels and title
    plt.xlabel('Time interval', fontsize=12)
    plt.ylabel('Testing Time (s)', fontsize=12)
    plt.xticks(data_nums)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # Setting the y-axis range
    plt.ylim(0, 0.1)  # Customize the range as needed

    # Adding a legend outside the plot at the top
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=3)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig('./plt/pred_test_time_dd.pdf')
    
    plt.show()
    
if __name__ == '__main__':
    data_nums, data, assign_data, train_time, test_time = parse_data()
    plot_data(data, data_nums)
    plot_assign(assign_data, data_nums)
    plot_train_time(train_time, data_nums)
    plot_test_time(test_time, data_nums)