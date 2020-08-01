import os

import cv2
import numpy as np

path = 'out'


def save(img, name):
    cv2.imwrite(os.path.join(path, name + ".jpg"), img)


if __name__ == '__main__':
    img = cv2.imread('clock.jpg')
    # 归一化
    nor = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)  # 按照比例缩放，如x,y轴均缩小一倍
    # save(nor, "nor")
    # 灰度
    gray = cv2.cvtColor(nor, cv2.COLOR_BGR2GRAY)
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
    save(binary, "bin")
    # 中值滤波
    median = cv2.medianBlur(binary, 1)
    # save(median, "median")
    # 高斯滤波
    gaussian = cv2.GaussianBlur(median, (3, 3), 0)
    # save(gaussian, "gaussian")
    # 边缘检测
    edges = cv2.Canny(gaussian, 60, 143, apertureSize=3)
    # 检测圆
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param2=200, minRadius=90)
    circles = np.uint16(np.around(circles))  # 把circles包含的圆心和半径的值变成整数
    cir = nor.copy()

    for i in circles[0, :]:
        cv2.circle(cir, (i[0], i[1]), i[2], (0, 255, 0), 2, cv2.LINE_AA)  # 画圆
        cv2.circle(cir, (i[0], i[1]), 2, (0, 255, 0), 2, cv2.LINE_AA)  # 画圆心
    # cv2.imshow("circles", cir)
    # save(cir, "circle")

    # 检测直线

    ig = cv2.GaussianBlur(cir, (3, 3), 0)
    # save(ig, "ig")

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    save(edges, 'edges')
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    if lines is None:
        print("检测不到直线")
    else:
        print(lines.shape)
