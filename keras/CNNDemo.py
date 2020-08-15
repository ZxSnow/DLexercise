from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten

model = Sequential()

# 25个filter 3*3结构
# input_shape 参数：1.图片类型（1代表黑白，3代表RGB）后两位代表像素大小为28*28
# 1x28x28
# 在次此处的Convolution中每个filter有9个参数（1*3*3）
model.add(Convolution2D(25, 3, 3, input_shape=(1, 28, 28)))

# 25x26x26
# 2，2为划分的单位
model.add(MaxPooling2D(2, 2))
# 25x13x13

# 在次此处的Convolution中每个filter有225个参数（25*3*3）
model.add(Convolution2D(50, 3, 3))
# 50x11x11

model.add(MaxPooling2D(2, 2))
# 50x5x5

model.add(Flatten())
# 1250维

model.add(Dense(output_dim=100, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))
