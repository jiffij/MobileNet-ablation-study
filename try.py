import matplotlib.pyplot as plt
import numpy as np

# Sample data
x = np.arange(10)
train_loss = np.random.random(10)
val_loss = np.random.random(10)
train_acc = np.random.random(10)
val_acc = np.random.random(10)

fig, ax1 = plt.subplots()
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
ax1.set_ylim([0, max(train_loss.max(), val_loss.max()) + 1])
lns1 = ax1.plot(x, train_loss, 'yo-', label='train_loss')
lns2 = ax1.plot(x, val_loss, 'go-', label='val_loss')

# Increase resolution on the y-axis
yticks = np.linspace(0, max(train_loss.max(), val_loss.max()) + 1, 20)
ax1.set_yticks(yticks)

ax2 = ax1.twinx()
ax2.set_ylabel('accuracy')
ax2.set_ylim([0, 1])
yticks = np.linspace(0, 1, 30)
ax2.set_yticks(yticks)
lns3 = ax2.plot(x, train_acc, 'bo-', label='train_acc')
lns4 = ax2.plot(x, val_acc, 'ro-', label='val_acc')

plt.show()