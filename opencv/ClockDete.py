import os

import cv2
import numpy as np

path = '../out'


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
    median = cv2.medianBlur(bin, 1)
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
            cv2.circle(cir, (i[0], i[1]), i[2], (0, 255, 0), 2, cv2.LINE_AA)  # 画圆
            cv2.circle(cir, (i[0], i[1]), 2, (0, 255, 0), 2, cv2.LINE_AA)  # 画圆心
        # cv2.imshow("circles", cir)
        save(cir, "circle")
        return cir, circles[0][0][0], circles[0][0][1], circles[0][0][2]


def dist_2_pts(x1, y1, x2, y2):
    # print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def cross_point(th1, r1, th2, r2):
    y1 = ((r2 * np.cos(th1)) - (r1 * np.cos(th2))) / ((np.sin(th2)) * (np.cos(th1)) - (np.sin(th1)) * (np.cos(th2)))
    print("y=", y)
    x1 = (r1 - y * np.sin(th1)) / np.cos(th1)
    x2 = (r2 - y * np.sin(th2)) / np.cos(th2)
    print("sin(th1)=", np.sin(th1), " ,cos(th1)=", np.cos(th1), " r1=", r1)
    print("sin(th2)=", np.sin(th2), " ,cos(th2)=", np.cos(th2), " r2=", r2)
    print("x1=", x1, ",x2=", x2)
    return x1, y1


def cross(points):
    k1, b1 = kb_value(points[0], points[1])
    k2, b2 = kb_value(points[2], points[3])
    x = (b2 - b1) / (k1 - k2)
    y = (k1 * x + b1)
    return x, y


def kb_value(point1, point2):
    (x1, y1), (x2, y2) = point1, point2
    k = (y1 - y2) / (x1 - x2)
    b = (y1 - k * x1)
    return k, b


def line_detection(cir, x, y):
    # 检测直线
    ig = cv2.GaussianBlur(cir, (3, 3), 0)
    save(ig, "ig")
    result = cir.copy()

    # 边缘检测
    edges = cv2.Canny(ig, 50, 150, apertureSize=3)
    save(edges, 'edges')

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    # lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=60,
    #                         maxLineGap=5)

    if lines is None:
        print("检测不到直线")
    else:
        print(lines.shape)

        # lines1 = lines[:, 0, :]
        #
        # for x1, y1, x2, y2 in lines1[:]:
        #     cv2.line(nor, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # save(nor, "demo")
        # line1 = lines[0]
        # line2 = lines[1]
        #
        # x_cross, y_cross = cross_point(line1[0][1], line1[0][0], line2[0][1], line2[0][0])
        # cv2.circle(result, (x_cross, y_cross), 1, (0, 255, 255), 2, cv2.LINE_AA)
        points = []

        # cv2.line(result, pt3, pt4, (0, 255, 0), 2, cv2.LINE_AA)
        for i in range(2):
            for line in lines[i]:
                rho = line[0]  # 第一个元素是距离rho
                theta = line[1]  # 第二个元素是角度theta
                rtheta = theta * (180 / np.pi)
                print('θ%d:' % i, rtheta)
                if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                    # 该直线与第一行的交点
                    pt1 = (int(rho / np.cos(theta)), 0)
                    # 该直线与最后一行的焦点
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    a = int(
                        int(int(rho / np.cos(theta)) + int(
                            (rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                    b = int(result.shape[0] / 2)
                    pt3 = (a, b)
                    pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))

                    points.append(pt3)
                    points.append(pt4)
                    # 绘制一条白线
                    # cv2.putText(result, 'theta1={}'.format(int(rtheta)), pt4, cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    #             (0, 0, 255), 1)
                    cv2.line(result, pt3, pt4, (0, 255, 0), 2, cv2.LINE_AA)
                else:  # 水平直线
                    # 该直线与第一列的交点
                    pt1 = (0, int(rho / np.sin(theta)))
                    # 该直线与最后一列的交点
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    a = int(
                        int(int(rho / np.cos(theta)) + int(
                            (rho - result.shape[0] * np.sin(theta)) / np.cos(theta))) / 2)
                    b = int(result.shape[0] / 2)
                    pt3 = (a, b)
                    pt4 = (int(int(int(rho / np.cos(theta)) + a) / 2), int(b / 2))
                    points.append(pt3)
                    points.append(pt4)
                    # 绘制一条直线
                    cv2.line(result, pt3, pt4, (0, 255, 0), 2, cv2.LINE_AA)
                    # cv2.imshow('Result', result)
        x_cross, y_cross = cross(points)
        cv2.circle(result, (int(x_cross), int(y_cross)), 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.line(result, (int(x_cross), int(y_cross)), (x, y), (0, 0, 255), 2, cv2.LINE_AA)
        save(result, "result")


def scale(cir):
    """
    在cir图上根据原区分刻度
    :param cir:
    :return: 返回刻度对应的像素值
    """
    separation = 6.0  # in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))  # set empty arrays
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    image = cir.copy()

    for i in range(0, interval):
        for j in range(0, 2):
            if j % 2 == 0:
                p1[i][j] = x + 0.9 * r * np.sin(separation * i * 3.14 / 180)  # point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.cos(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if j % 2 == 0:
                p2[i][j] = x + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.sin(
                    (separation) * (i + 0) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2 * r * np.cos(
                    (separation) * (i + 0) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
    for i in range(0, interval):
        cv2.line(image, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(image, '%s' % (int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 0), 1, cv2.LINE_AA)
    save(image, "scale")
    return p2, image


def scale_pix(value):
    x_, y_ = x + r * np.sin(value * 3.14 / 180), y + r * np.cos(value * 3.14 / 180)
    return x_, y_


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

    cir, x, y, r = circle_detection(binary, nor)
    print("检测圆相关信息", x, y, r)

    pixes, scale = scale(cir)

    min = 312
    max = 46
    x_min, y_min = scale_pix(min)
    x_max, y_max = scale_pix(max)
    cv2.circle(scale, (int(x_min), int(y_min)), 3, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(scale, (int(x_max), int(y_max)), 3, (0, 0, 255), 2, cv2.LINE_AA)
    save(scale, "scale-pix")

    line_detection(cir, x, y)
