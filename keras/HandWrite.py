"""
    keras教程：手写数字识别（李宏毅视频）
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

# 数据集
f = np.load('../file/mnist.npz')
(X_train, y_train) = f['x_train'], f['y_train']
(X_test, y_test) = f['x_test'], f['y_test']
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
print("数据加载完成")

# 数据预处处理
# 数据标准化
X_train = X_train.reshape(X_train.shape[0], -1) / 255.
X_test = X_test.reshape(X_test.shape[0], -1) / 255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 为测试集添加噪声，用以演示dropout
X_test = np.random.normal(X_test)

# 1. 构建模型
model = Sequential()

# 逐层构造
# 第一层
# model.add(Dense(input_dim=28 * 28, output_dim=500))
# model.add(Activation('sigmoid'))
model.add(Dense(input_dim=28 * 28, units=500, activation='relu'))
model.add(Dropout(0.7))

# 第二层
# model.add(Dense(output_dim=500))
# model.add(Activation('sigmoid'))
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.7))

# 第三层
# model.add(Dense(units=500, activation='relu'))
# model.add(Dropout(0.7))

# 输出层
# model.add(Dense(output_dim=10))
# model.add(Activation('softmax'))
model.add(Dense(units=10, activation='softmax'))

# 2. 评价模型好坏
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#  训练网络
model.fit(X_train, y_train, epochs=4, batch_size=100)

loss, accuracy = model.evaluate(X_train, y_train)
print('train loss: ', loss)
print('train accuracy: ', accuracy)

# 评价训练出的网络
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)
