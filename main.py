import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt

def main():
    x_train, y_train, processor_speed, ram_size, storage_capacity = read_csv_file()
    w_in = np.array([0., 0., 0., 0.])
    b_in = 0.
    iterations = 1500
    alpha = 0.000003

    w, b, J_hist, num_iters = gradient_descent(x_train, y_train, w_in, b_in, compute_cost, compute_gradient, alpha, iterations)
    print("Final w and b values are : ", w, b)

    print(f"The model's prediction is : {predict(w, b) * 10 ** 3}")

    show_J_hist(J_hist, num_iters)
    show_features_data(processor_speed, ram_size, storage_capacity)
    show_price_data(processor_speed, ram_size, y_train)
    show_model(processor_speed, ram_size, y_train, x_train, w, b)

def read_csv_file():
    df = pd.read_csv("Laptop_price.csv")
    processor_speed = df["Processor_Speed"][:100]
    ram_size = df["RAM_Size"][:100]
    storage_capacity = df["Storage_Capacity"][:100]
    screen_size = df["Screen_Size"][:100]
    prices = df["Price"][:100]
    price = [price / 1000 for price in prices]
    x_train = []

    for i in range(len(ram_size)):
        example = [processor_speed[i], ram_size[i], storage_capacity[i], screen_size[i]]
        x_train.append(example)

    x_train = np.array(x_train)
    y_train = np.array(price)

    # x_train = zscore_normalize_features(x_train)

    return x_train, y_train, processor_speed, ram_size, storage_capacity

def compute_cost(X, y, w, b):
    total_cost = 0
    m = X.shape[0]
    cost = 0

    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        err_i = (f_wb_i - y[i]) ** 2
        cost += err_i
    
    total_cost = cost * (1 / (2 * m))

    return total_cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err_i = (np.dot(X[i], w) + b) - y[i]

        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]

        dj_db = dj_db + err_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, cost_func, gradient_func, alpha, num_iters):
    J_hist = []
    num_iters_list = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_func(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_hist.append(cost_func(X, y, w, b))
            num_iters_list.append(int(i))
        
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_hist[-1]:8.2f}")

    return w, b, J_hist, num_iters_list

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)

    X_norm = (X - mu) / sigma

    return X_norm

def show_J_hist(J_hist, num_iters):
    plt.plot(num_iters[0:10], J_hist[0:10])
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()

def show_features_data(processor_speed, ram_size, storage_capacity):
    ax = plt.axes(projection='3d')

    ax.set_title("CPU - RAM - SSD")
    ax.set_xlabel("Processor speed")
    ax.set_ylabel("Ram size")
    ax.set_zlabel("Storage Capacity")

    ax.scatter3D(processor_speed, ram_size, storage_capacity)
    plt.show()

def show_price_data(processor_speed, ram_size, price):
    ax = plt.axes(projection='3d')

    ax.set_title("CPU - RAM - PRICE")
    ax.set_xlabel("Processor speed")
    ax.set_ylabel("Ram size")
    ax.set_zlabel("Price")

    ax.scatter3D(processor_speed, ram_size, price)
    plt.show()

def show_model(processor_speed, ram_size, price, x_train, w, b):
    ax = plt.axes(projection='3d')
    prices = np.dot(x_train, w) + b

    ax.set_title("CPU - RAM - PRICE")
    ax.set_xlabel("Processor speed")
    ax.set_ylabel("Ram size")
    ax.set_zlabel("Price")

    ax.scatter3D(processor_speed, ram_size, prices)

    ax.scatter3D(processor_speed, ram_size, price)
    plt.show()


def predict(w, b):
    processor_speed = float(input("Please enter the processor speed (in HZs) : "))
    ram_size = float(input("Please enter the RAM size (in GBs) : "))
    storage_capacity = float(input("Please enter the storage capacity (in GBs) : "))
    screen_size = float(input("Please enter the screen size (in inchs) : "))
    pred_inp = [processor_speed, ram_size, storage_capacity, screen_size]
    pred_inp = np.array(pred_inp)

    prediction = np.dot(pred_inp, w) + b

    return prediction

main()