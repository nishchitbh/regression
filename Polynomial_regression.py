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
    w0 = random.randint(0,4)
    w1 = random.random()
    b = random.random()
    learning_rate = 0.00000293
    y_ = w0*(x**2)+w1*x+b
    stats = {'w0': [w0], 'w1': [w1], 'b': [b], 'J': [np.sum((y_-y)**2)/(2*m)]}
    for i in range(146400):
        tmp_w0 = w0 - learning_rate * np.sum((y_-y)*(x**2))/m
        tmp_w1 = w1 - learning_rate * np.sum((y_-y)*x)/m
        tmp_b = b - learning_rate * np.sum(y_-y)/m
        w0 = tmp_w0
        w1 = tmp_w1
        b = tmp_b
        y_ = w0*(x**2)+w1*x+b
        J = np.sum((y_-y)**2)
        J = J/(2*m)
        stats['w0'].append(w0)
        stats['w1'].append(w1)
        stats['b'].append(b)
        stats['J'].append(J)
    return stats


# Running program
location = '../dataset-maker/dataset.json'
load_data = dataset(location)
x = load_data['x']
y = load_data['y']
m = load_data['m']
stats = gradient_descent(x, y, m)
init_w0 = stats['w0'][0]
init_w1 = stats['w1'][0]
init_b = stats['b'][0]
w0 = stats['w0'][-1]
w1 = stats['w1'][-1]
b = stats['b'][-1]
print('Min cost', min(stats['J']))
print('Initial w:', init_w0)
print('Initial w1:',init_w1)
print('Final w0:', w0)
print('Final w1:',w1)
print('Initial b:', init_b)
print('Final b', b)

# Testing Accuracy
accuracy = 0
for i in range(m):
    calculated_y = w0*(x[i]**2)+w1*x[i]+b
    accuracy += abs(calculated_y-y[i])*100/y[i]
accuracy = 100-accuracy/m
print(f"Accuracy: {accuracy}%")

# Data Visualization
plx = np.linspace(0, 45, m)
ply = np.polyval([w0,w1,b], plx)
ply_init = init_w0*plx+init_b
plt.figure(1)
plt.xlabel("w", color='red')
plt.ylabel("J", color='red')
plt.scatter(stats['w0'], stats['J'])
plt.scatter(stats['w0'][-1],stats['J'][-1], color='red')
plt.title('Gradient descent')

plt.figure(2)
plt.xlabel('x', color='red')
plt.ylabel('y', color='red')
plt.plot(plx, ply_init, color='blue', label='Before Polynomial Regression')
plt.plot(plx, ply, color='red', label='After Polynomial Regression')
plt.scatter(x, y)
plt.legend()
plt.show()
