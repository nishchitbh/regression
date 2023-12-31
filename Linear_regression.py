import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import random
# Extracting data


def dataset(location):
    dset = np.array(json.load(open(location))['red'])
    return {'m': len(dset), 'x': dset[:, 0], 'y': dset[:, 1]}

# Gradient Descent


def gradient_descent(x, y, m):
    w = random.random()
    b = random.random()
    init_w = w
    init_b = b
    learning_rate = 0.002
    y_ = w*x+b
    stats = {'w': [w], 'b': [b], 'J': [np.sum((y_-y)**2)/(2*m)]}
    for i in range(125400):
        tmp_w = w - learning_rate * np.sum((y_-y)*x)/m
        tmp_b = b - learning_rate * np.sum(y_-y)/m
        w = tmp_w
        b = tmp_b
        y_ = w*x+b
        J = np.sum((y_-y)**2)/(2*m)
        stats['w'].append(w)
        stats['b'].append(b)
        stats['J'].append(J)
    return stats


# Running program
location = "../dataset-maker/dataset.json"
load_data = dataset(location)
x = load_data['x']
y = load_data['y']
m = load_data['m']
stats = gradient_descent(x, y, m)
init_w = stats['w'][0]
init_b = stats['b'][0]
w = stats['w'][-1]
b = stats['b'][-1]
print('Min cost', min(stats['J']))
print('Initial w:', init_w)
print('Final w:', w)
print('Initial b:', init_b)
print('Final b', b)

# Testing Accuracy
accuracy = 0
for i in range(m):
    calculated_y = w*x[i]+b
    accuracy += abs(calculated_y-y[i])*100/y[i]
accuracy = 100-accuracy/m
print(f"Accuracy: {accuracy}%")

# Data Visualization
plx = np.linspace(0, 45, m)
ply = w*plx+b
ply_init = init_w*plx+init_b
plt.figure(1)
plt.xlabel("w", color='red')
plt.ylabel("J", color='red')
plt.scatter(stats['w'], stats['J'])
plt.title('Gradient descent')

plt.figure(2)
plt.xlabel('x', color='red')
plt.ylabel('y', color='red')
plt.plot(plx, ply_init, color='blue', label='Before Linear Regression')
plt.plot(plx, ply, color='red', label='After Linear Regression')
plt.scatter(x, y)
plt.legend()
plt.show()
