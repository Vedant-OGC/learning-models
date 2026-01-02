import numpy as np
import matplotlib.pyplot as plt


def loss(w):
    return w**2 + 4*w + 5  # Minimum at w = -2


def grad(w):
    return 2*w + 4

w = 5           # starting point
lr = 0.1        # learning rate
steps = 20

history = []

# Gradient Descent loop
for i in range(steps):
    history.append(w)
    w = w - lr * grad(w)


w_vals = np.linspace(-5, 5, 200)
loss_vals = loss(w_vals)

plt.figure(figsize=(8,5))
plt.plot(w_vals, loss_vals, label="Loss function")
plt.scatter(history, loss(np.array(history)), color='red', label="Steps")
plt.title("Gradient Descent Visualization")
plt.xlabel("w (weight)")
plt.ylabel("Loss")
plt.legend()
plt.show()
