import math
import random
import time
import matplotlib.pyplot as plt

def mean_and_std(x):
    n = len(x)
    mean = sum(x) / n
    squared_diff = [(element - mean) ** 2 for element in x]
    sample_std = math.sqrt(sum(squared_diff) / n)
    return mean, sample_std
# x = [1, 2, 3, 4, 5]
# result = mean_and_std(x)
# print("The mean and standard deviation of ", x, " is", result)

def normalize(filename):
    file = f = open(filename, 'r')
    header = file.readline().strip()
    data_list = []
    for line in file:
        area, bedrooms, price = map(float, line.strip().split(','))
        data_list.append([area, bedrooms, price])
    
    mean_area, std_area = mean_and_std([row[0] for row in data_list])
    mean_bedrooms, std_bedrooms = mean_and_std([row[1] for row in data_list])
    mean_price, std_price = mean_and_std([row[2] for row in data_list])
    
    # Normalize the data
    normalized_data_list = []
    for area, bedrooms, price in data_list:
        normalized_area = (area - mean_area) / std_area
        normalized_bedrooms = (bedrooms - mean_bedrooms) / std_bedrooms
        normalized_price = (price - mean_price) / std_price
        normalized_data_list.append([normalized_area, normalized_bedrooms, normalized_price])
    # print(normalized_data_list)

    # Write the normalized data to normalized.txt
    with open('normalized.txt', 'w') as file:
        file.write(header + '\n')
        for normalized_area, normalized_bedrooms, price in normalized_data_list:
            file.write(f"{normalized_area},{normalized_bedrooms},{price}\n")

    return normalized_data_list, mean_area, mean_bedrooms, mean_price, std_area, std_bedrooms, std_price

def loss_function(w, data):
    sum_squares = 0
    for area, bedrooms, price in data:
        prediction = w[0] + w[1]*area + w[2]*bedrooms
        sum_squares += (prediction - price) ** 2
    return (1/(2 * len(data))) * sum_squares

def gradient_descent(initial_w, alpha, data):
    w = initial_w.copy()
    cycle_count = 0
    m = len(data)
    losses = []
    for cycle_count in range(80): # cycle_count counts completed cycles
        w_temp = w.copy()
        for i in range(len(w)):
            sum = 0
            for area, bedrooms, price in data:
                x_j = [1, area, bedrooms][i]  # 1, area, or bedrooms depending on i
                prediction = w[0] + w[1] * area + w[2] * bedrooms
                sum += (prediction - price) * x_j
            w_temp[i] = w[i] - alpha * (1 / m) * sum
        w = w_temp
        if ((cycle_count+1) % 10 == 0): # If loss is of interest log it
            print("Cycle #" + str(cycle_count+1) + " Loss:", loss_function(w, data))
            loss_value = loss_function(w, data)
            losses.append((cycle_count + 1, loss_value))
    # print("Total # of Cycles (Alpha = " + str(alpha) + "):", cycle_count+1)
    return w, losses

def stochastic_gradient_descent(initial_w, alpha, data):
    w = initial_w.copy()
    for cycle in range(3):
        for area, bedrooms, price in data:
            w_temp = w.copy()
            for i in range(len(w)):
                x_j = [1, area, bedrooms][i]  # 1, area, or bedrooms depending on i
                prediction = w[0] + w[1] * area + w[2] * bedrooms
                w_temp[i] = w[i] - alpha * (prediction - price) * x_j # no 1/m?
            w = w_temp
        random.shuffle(data)
        print("Stochastic loss after cycle " + str(cycle + 1)+ ":", loss_function(w, data))
    return w


normalized_data_list, mean_area, mean_bedrooms, mean_price, std_area, std_bedrooms, std_price = normalize("housing.txt")
# print (normalized_data_list)
w = [0, 0, 0]
# print("Loss function", loss_function(w, normalized_data_list))

# Collect loss values for different alphas
alphas = [0.01, 0.03, 0.1, 0.2, 0.5, 0.05]
w_by_alpha = []
losses_by_alpha = {}
gradient_times = []
for alpha in alphas:
    print("\nTesting alpha = " + str(alpha) + "!")
    gradient_start_time = time.time()
    w, losses = gradient_descent([0, 0, 0], alpha, normalized_data_list)
    gradient_end_time = time.time()
    gradient_time_elapsed = gradient_end_time - gradient_start_time
    gradient_times.append(gradient_time_elapsed)
    print("Gradient Time Elapsed (alpha = " + str(alpha) + "):", gradient_time_elapsed)
    w_by_alpha.append(w)
    losses_by_alpha[alpha] = losses

# Plotting
plt.figure(figsize=(12, 8))
for alpha, losses in losses_by_alpha.items():
    cycles, loss_values = zip(*losses)
    plt.plot(cycles, loss_values, label=f"Alpha = {alpha}")

plt.xlabel('# of Cycles')
plt.ylabel('Loss Value')
plt.title('Loss Values for Different Alphas')
plt.legend()
plt.grid(True)

print("Testing stochastic gradient descent for alpha = " + str(0.05) + "!")
stochastic_start_time = time.time()
w_stochastic = stochastic_gradient_descent([0, 0, 0], 0.05, normalized_data_list)
stochastic_end_time = time.time()
stochastic_time_elapsed = stochastic_end_time - stochastic_start_time
print("Stochastic Time Elapsed:", stochastic_time_elapsed)

# Part C answered with the best observed learning rate --> 0.5
w, losses = gradient_descent([0, 0, 0], 0.5, normalized_data_list)
normalized_price = w[0] + w[1]*(2650-mean_area)/std_area + w[2]*(4-mean_bedrooms)/std_bedrooms
predicted_price = (normalized_price * std_price) + mean_price
print("Predicted price of a house w/ 2650 square feet and 4 bedrooms:", predicted_price)
print ("w:", w)

plt.show()