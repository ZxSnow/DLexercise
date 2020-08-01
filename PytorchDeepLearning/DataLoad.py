import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from PytorchDeepLearning.multiple import MultipleModel

'''Dataset是一个抽象类'''


# filepath = "database/diabetes.csv.gz"


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        """常见策略
           1. 小数据量，可以将所有的data都读进内存，然后在getitem中逐个返回
           2. 数据量较大时，可能需要按照文件来进行分块的读取"""
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        """支持下标操作"""
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """返回数据长度"""
        return self.len


dataset = DiabetesDataset("database/diabetes.csv.gz")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

model = MultipleModel()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        # 1. 数据准备
        inputs, labels = data
        # 2. 前向
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.item())

        # 3. backward
        optimizer.zero_grad()
        loss.backward()

        # 4. 更新
        optimizer.step()
