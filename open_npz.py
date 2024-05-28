import numpy as np

data = np.load('./diagram/lr=0.05-.png.npz', allow_pickle=True)
lst = data.files

for item in lst:
    print(item)
    print(data[item])