import rudygrad
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

print("Generating moons dataset...")
x_data, y_data = make_moons(n_samples=100, noise=0.05)

# sklearn y_data is 0 and 1, we want -1.0 and 1.0 for our margin loss
y_data = np.where(y_data == 0, -1.0, 1.0)

inputs = [[rudygrad.Value(x) for x in row] for row in x_data]
targets = [rudygrad.Value(y) for y in y_data]

model = rudygrad.MLP(2, [16, 16, 1])

start_time = time.time()
core_time = 0

for k in range(101):
    step_start = time.time()
    total_loss = rudygrad.Value(0.0)
    correct = 0

    for x, y in zip(inputs, targets):
        score = model.call(x)[0]
        # margin = 1.0 - (y * score)
        margin = rudygrad.Value(1.0) - (y * score)
        loss = margin.relu()
        total_loss = total_loss + loss

        score_val = score.data
        y_val = y.data
        if (y_val > 0.0 and score_val > 0.0) or (y_val < 0.0 and score_val < 0.0):
            correct += 1

    final_loss = total_loss * rudygrad.Value(1.0 / len(inputs))
    accuracy = (correct / len(inputs)) * 100.0

    model.zero_grad()
    final_loss.backward()

    learning_rate = 1.0 - (0.9 * k / 100.0)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    core_time += time.time() - step_start

    if k % 5 == 0:
        print(f"Step {k:>3} | Loss: {final_loss.data:.4f} | Accuracy: {accuracy:.1f}%")

print(f"\nTraining complete!")
print(f"Total training duration:  {time.time() - start_time:.4f}s")
print(f"Core training time only: {core_time:.4f}s")
print(f"Average core step time:  {(core_time / 101.0) * 1000.0:.4f}ms")

# Plot final decision boundary
print("\nPlotting decision boundary...")
res = 60
xs = np.linspace(-1.5, 2.5, res)
ys = np.linspace(-1.0, 1.5, res)
xx, yy = np.meshgrid(xs, ys)
Z = np.zeros_like(xx)

for i in range(res):
    for j in range(res):
        pt = [rudygrad.Value(xx[i, j]), rudygrad.Value(yy[i, j])]
        score = model.call(pt)[0].data
        Z[i, j] = np.tanh(score)

plt.contourf(xx, yy, Z, cmap='bwr', alpha=1, vmin=-1.0, vmax=1.0, levels=np.linspace(-1, 1, 100))
plt.scatter([x[0] for x in x_data], [x[1] for x in x_data], c=y_data, cmap='bwr', vmin=-1.0, vmax=1.0, edgecolors='k')
plt.savefig("demo.png")
print("Saved plot to 'demo.png'")
