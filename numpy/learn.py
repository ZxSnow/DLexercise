import numpy as np

'''
(axis=0 表示对列进行操作；axis=1表示对行进行操作)
基础学习：
1.矩阵定义
2.特殊矩阵定义
3.矩阵运算（逐个相乘、矩阵乘法、矩阵内求和、矩阵平均值，中位数、矩阵转置）
4.矩阵内索引
5.array合并与分割
6.np.array 在使用等号赋值的时候，多个变量会被同步修改（是包含地址的深拷贝），不想同步更改可以使用 b = a.copy()方法来浅拷贝
'''

# 矩阵定义
array = np.array([[1, 2], [4, 5]])
# 矩阵参数信息
print(array)
print('维度:', array.ndim)
print('行*列:', array.shape)
print('大小:', array.size)

# 带格式定义矩阵
# np.int 默认为64位 (可添加位数)
# np.int np.float
array2 = np.array([[2, 3], [4, 5]], dtype=np.int)
print(array2.dtype)

# 特殊矩阵定义
array3 = np.zeros((3, 3))
print(array3)
array4 = np.ones((4, 5), dtype=np.int)
print(array4)

array5 = np.arange(4).reshape((2, 2))
print(array5)
# 逐个相乘
print(array * array5)
# 矩阵乘法
c = np.dot(array, array5)
print(c)

a = np.random.random((2, 4))
# 矩阵内求和
np.sum(a)
# 矩阵内求最值 axis=0在列内，axis=1 在行内 无axis 在矩阵内
print(a)
print(np.max(a, axis=0))
np.min(a)

A = np.arange(2, 14).reshape((3, 4))
# 显示矩阵中最小值所在的坐标、最大值
np.argmin(A)
np.argmax(A)

# 矩阵的平均值
np.mean(A)
np.average(A)
A.mean()

# 中位数
np.median(A)

# 逐步累积
np.cumsum(A)

# 相邻求差
np.diff(A)

# 非零数
np.nonzero(A)

# 排序
np.sort(A)

# A的转置
np.transpose(A)
A_T = A.T
print(A_T)

# 截断 小于5的都设置为5，大于10的都设为10
np.clip(A, 5, 10)

# 索引
# 从零还是计数
print(A[2])
print(A[2, :])
# 索引单个元素
print(A[2][1])
print(A[2, 1])
# 冒号的使用
print(A[2, :])
print(A[:, 1])
# 表示从1到3，但不包含三
print(A[1, 1:3])

# 循环矩阵的每行
for row in A:
    print(row)

# 循环每列
for column in A.T:
    print(column)

# 循环每一个元素 A.flat返回一个迭代器
# A.flatten() 返回一个数组
for item in A.flat:
    print(item)

''' array合并 此时定义的A是没有维度的，可以通过newaxis添加为行或列添加维度 '''
A0 = np.array([1, 1, 1])
# A1 = [[1 1 1]] （一维行向量）
A1 = A0[np.newaxis, :]
# A2 = [[1][1][1]] (一维列向量)
A2 = A0[:, np.newaxis]
B = np.array([2, 2, 2])
# 上下合并
C = np.vstack((A0, B))
# 左右合并
D = np.hstack((A0, B))

# 可指定方向的合并 axis=0 上下堆叠; axis=1 左右堆叠
np.concatenate((A2, B[:, np.newaxis], A2), axis=0)

''' array 分割 '''
# split是等量分割，不支持进行不等量分割
# 横向分成三份 认为将列进行分割
print(np.split(A, 3, axis=0))
np.vsplit(A, 3)
# 纵向分成两份 认为将行进行分割
print(np.split(A, 2, axis=1))
np.hsplit(A, 2)
# 不等量分割
print(np.array_split(A, 2, axis=1))

