import os

import cv2
import numpy as np

path = 'out'


def save(img, name):
    cv2.imwrite(os.path.join(path, name + ".jpg"), img)


def normalize(img):
    # 归一化 输入尺寸为3000*4000
    return cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)  # 按照比例缩放，如x,y轴均缩小一倍


def gray(nor):
    # 灰度
    return cv2.cvtColor(nor, cv2.COLOR_BGR2GRAY)


def threshold(gray):
    _, binary = cv2.threshold(gray, 80, 255, 0)
    return binary


def filters(bin):
    median = cv2.medianBlur(binary, 1)
    save(median, "median")
    # 高斯滤波
    gaussian = cv2.GaussianBlur(median, (3, 3), 0)
    save(gaussian, "gaussian")

    return gaussian


def circle_detection(gray, img):
    # 检测圆
    height, width = img.shape[:2]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, param2=50, minDist=int(height * 0.35),
                               maxRadius=int(height * 0.5))
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 100, param2=200, maxRadius=380)
    if circles is None:
        print("未检测到圆形，参数不正确")
    else:
        print("检测到圆形")
        circles = np.uint16(np.around(circles))  # 把circles包含的圆心和半径的值变成整数
        cir = img.copy()

        # 为了只画了一个圆
        for i in circles[0][0:1]:
            cv2.circle(cir, (i[0], i[1]), i[2], (155, 50, 255), 2, cv2.LINE_AA)  # 画圆
            cv2.circle(cir, (i[0], i[1]), 2, (155, 50, 255), 2, cv2.LINE_AA)  # 画圆心
        # cv2.imshow("circles", cir)
        save(cir, "circle")
        return cir


def line_detection(cir):
    # 检测直线
    ig = cv2.GaussianBlur(cir, (3, 3), 0)
    save(ig, "ig")
    result = cir.copy()

    # 边缘检测
    edges = cv2.Canny(ig, 50, 150, apertureSize=3)
    save(edges, 'edges')

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 118)
    if lines is None:
        print("检测不到直线")
    else:
        print(lines.shape)
        for line in lines[0]:
            rho = line[0]  # 第一个元素是距离rho
            theta = line[1]  # 第二个元素是角度theta
            rtheta = theta * (180 / np.pi)
            print('θ1:', rtheta)
            if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                # 该直线与第一行的交点
                pt1 = (int(rho / np.cos(theta)), 0)
                # 该直线与最后一行的焦点
                pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                a = int(
                    int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                b = int(result.shape[0] / 2)
                pt3 = (a, b)
                pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
                # 绘制一条白线
                cv2.putText(result, 'theta1={}'.format(int(rtheta)), pt4, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                cv2.line(result, pt3, pt4, (0, 0, 255), 2, cv2.LINE_AA)
            else:  # 水平直线
                # 该直线与第一列的交点
                pt1 = (0, int(rho / np.sin(theta)))
                # 该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                a = int(
                    int(int(rho / np.cos(theta)) + int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                b = int(result.shape[0] / 2)
                pt3 = (a, b)
                pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
                # 绘制一条直线
                cv2.line(result, pt3, pt4, (0, 0, 255), 2, cv2.LINE_AA)
                # cv2.imshow('Result', result)
            save(result, "result")


if __name__ == '__main__':
    num = input("demo:")
    img = cv2.imread('pic/demo-%s.jpg' % num)

    nor = normalize(img)
    save(nor, "nor")

    # 灰度
    gray = gray(nor)
    save(gray, "gray")
    # 二值化
    binary = threshold(gray)
    save(binary, "bin")
    # 滤波
    gaussian = filters(binary)

    # # 边缘检测
    # edges = cv2.Canny(gaussian, 60, 143, apertureSize=3)
    # save(edges, "edges")

    cir = circle_detection(binary, nor)

    line_detection(cir)
