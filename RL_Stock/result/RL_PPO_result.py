"""
fish
conda activate mlntu
cd result
python RL_PPO_result.py
"""

import pickle
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    moving_averages = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        average = sum(window) / window_size
        moving_averages.append(average)
    return moving_averages


model_name = 'resnet'
method = "PPO"
short = "_short"
# epoch = "_3000" # if "": 1000
with open(f'{method}/{model_name}/returns_train_val{short}.pkl', 'rb') as f:
    (returns_train, returns_val, loss_train) = pickle.load(f)
window_size = 50
plt.figure(figsize=(17, 6))
plt.subplot(1, 2, 1)
plt.plot(returns_train, label='train rewards')
plt.plot(returns_val, label='val rewards')
plt.xlabel('epoch')
plt.ylabel('rewards')
plt.subplot(1, 2, 2)
plt.plot(loss_train, label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
with open(f'{method}/{model_name}/returns_train_val{short}_backup.pkl', 'wb') as f:
    pickle.dump((returns_train, returns_val, loss_train), f)
