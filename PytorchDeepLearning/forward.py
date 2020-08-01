import torch
""" Tensor在进行加法运算的时候会构建计算图，因为在进行纯数值计算的时，应落实到data（或item） """

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True


def forward(x):
    return x * w


# 理解为创造一个新的计算图
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict before training:", 4, forward(4).item())

for epoch in range(100):
    for (x, y) in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print("\tgrad", x, y, w.grad.item())
        # 进行纯数值修改
        w.data = w.data - 0.01 * w.grad.data

        # 对积分进行清零，否则下一次的计算会被累加
        w.grad.data.zero_()

    print("process", epoch, l.item())

print("predict before training:", 4, forward(4).item())
