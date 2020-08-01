import torch

in_channel, out_channel = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size, in_channel, width, height)

conv_layer = torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size)

output = conv_layer(input)
