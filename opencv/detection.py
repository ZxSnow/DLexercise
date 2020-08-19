import cv2
import os
import numpy as np
import math


class Detection(object):
    def __init__(self, path):
        self.meter = None
        self.img = None
        self.x = 0
        self.y = 0
        self.r = 0
        self.path = path
        self.output = 'out'
        self.min_p = (0, 0)
        self.max_p = (0, 0)
        self.cross_p = (0, 0)
        # 读取图片

    # 图片剪裁
    def cut(self):
        img = cv2.imread(self.path)
        meter = self.meter
        cropped = img[meter['y0']:meter['y1'], meter['x0']:meter['x1']]
        self.img = cropped
        # self.save(cropped, "cut")

    # 图片归一化处理
    def normalized_pic(self):
        nor = cv2.resize(self.img, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)  # 按照比例缩放，如x,y轴均缩小一倍
        # cv2.imshow('Normalized picture', nor)
        return nor

    def save(self, img, name):
        cv2.imwrite(os.path.join(self.output, "%s-%s.jpg" % (name, self.meter['idx'])), img)

    # 颜色空间转换：灰度化
    def color_gray(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
        return img_gray

    def threshold(self, gray):
        _, binary = cv2.threshold(gray, 80, 255, 0)
        return binary

    def cross(self, points):
        k1, b1 = self.kb_value(points[0], points[1])
        k2, b2 = self.kb_value(points[2], points[3])
        x = (b2 - b1) / (k1 - k2)
        y = (k1 * x + b1)
        return x, y

    def kb_value(self, point1, point2):
        (x1, y1), (x2, y2) = point1, point2
        k = (y1 - y2) / (x1 - x2)
        b = (y1 - k * x1)
        return k, b

    def circle_detection(self, gray, img):
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
            self.save(cir, "circle")
            self.x = circles[0][0][0]
            self.y = circles[0][0][1]
            self.r = circles[0][0][2]
            return cir, circles[0][0][0], circles[0][0][1], circles[0][0][2]

    def line_detection(self, cir, x, y):
        # 检测直线
        ig = cv2.GaussianBlur(cir, (3, 3), 0)
        # self.save(ig, "ig")
        result = cir.copy()

        # 边缘检测
        edges = cv2.Canny(ig, 50, 150, apertureSize=3)
        # self.save(edges, 'edges')

        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        # lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=60,
        #                         maxLineGap=5)

        if lines is None:
            print("检测不到直线")
        else:
            print(lines.shape)

            points = []

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
            # 两线交点,range(2)时使用
            x_cross, y_cross = self.cross(points)
            cv2.circle(result, (int(x_cross), int(y_cross)), 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(result, (int(x_cross), int(y_cross)), (x, y), (0, 0, 255), 2, cv2.LINE_AA)
            self.save(result, "result")
            self.cross_p = (x_cross, y_cross)
            return x_cross, y_cross

    def scale(self, cir):
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
        x = self.x
        y = self.y
        r = self.r

        for i in range(0, interval):
            for j in range(0, 2):
                if j % 2 == 0:
                    p1[i][j] = x + 0.9 * r * np.sin(separation * i * 3.14 / 180)  # point for lines
                else:
                    p1[i][j] = y + 0.9 * r * np.cos(separation * i * 3.14 / 180)
        text_offset_x = 8
        text_offset_y = 3
        for i in range(0, interval):
            for j in range(0, 2):
                if j % 2 == 0:
                    p2[i][j] = x + r * np.sin(separation * i * 3.14 / 180)
                    p_text[i][j] = x - text_offset_x + 1.15 * r * np.sin(
                        (separation) * (
                                i + 0) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
                else:
                    p2[i][j] = y + r * np.cos(separation * i * 3.14 / 180)
                    p_text[i][j] = y + text_offset_y + 1.15 * r * np.cos(
                        (separation) * (
                                i + 0) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees
        for i in range(0, interval):
            cv2.line(image, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
            cv2.putText(image, '%s' % (int(i * separation)), (int(p_text[i][0]), int(p_text[i][1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 1, cv2.LINE_AA)
        self.save(image, "scale")
        return p2, image

    def scale_pix(self, value):
        x = self.x
        y = self.y
        r = self.r
        x_, y_ = x + r * np.sin(value * 3.14 / 180), y + r * np.cos(value * 3.14 / 180)
        return x_, y_

    def draw_point(self, value, img=None):
        x_, y_ = self.scale_pix(value)
        if img is not None:
            cv2.circle(img, (int(x_), int(y_)), 3, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(img, (self.x, self.y), (int(x_), int(y_)), (0, 255, 0), 2)

            self.save(img, "scale")
        print(value, "点坐标：", x_, y_)
        return x_, y_

    def filters(self, bin):
        # 中值滤波
        median = cv2.medianBlur(bin, 1)
        # 高斯滤波
        gaussian = cv2.GaussianBlur(median, (3, 3), 0)
        return gaussian

    def read(self):
        """
        对仪表进行初步读取
        读取到表盘以及指针,产生circle result两个图
        :return:
        """
        nor = self.normalized_pic()
        gray = self.color_gray(nor)
        bin = self.threshold(gray)
        fiter = self.filters(bin)

        cir, x, y, r = self.circle_detection(bin, nor)
        # 检测直线
        self.line_detection(cir, x, y)

    def angle(self, v1, v2):
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
        angle1 = math.atan2(dy1, dx1)
        angle1 = angle1 * 180 / math.pi
        # print(angle1)
        angle2 = math.atan2(dy2, dx2)
        angle2 = angle2 * 180 / math.pi
        # print(angle2)
        if angle1 * angle2 >= 0:
            included_angle = abs(angle1 - angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle
        return included_angle

    def measure(self):
        max_ = self.meter['max_angle']
        min_ = self.meter['min_angle']
        x_min, y_min = self.draw_point(int(min_))
        x_max, y_max = self.draw_point(int(max_))
        self.min_p = (x_min, y_min)
        self.max_p = (x_max, y_max)

    def scale_write(self):
        nor = self.normalized_pic()
        gray = self.color_gray(nor)
        bin = self.threshold(gray)

        cir, x, y, r = self.circle_detection(bin, nor)
        self.scale(cir)

        self.measure()

        (x_min, y_min) = self.min_p
        (x_max, y_max) = self.max_p
        v1 = [x, y, x_min, y_min]
        v2 = [x, y, x_max, y_max]
        du = self.angle(v1, v2)
        print("夹角度数：", du)
        list = {}
        file = open('../conf/mete-%s.txt' % self.meter['idx'], 'w')
        sub = int(self.meter['max_value']) - int(self.meter['min_value'])
        for i in range(int(360 - du) + 1):
            va = 1.6 * i / int(360 - du)
            list[i] = va
        file.write(str(list))
        file.close()

    def read_value(self):
        self.read()
        self.measure()

        v1 = [self.x, self.y, self.min_p[0], self.min_p[1]]
        v2 = [self.x, self.y, self.cross_p[0], self.cross_p[1]]
        angle = self.angle(v1, v2)
        f = open('../conf/mete-%s.txt' % self.meter['idx'], 'r')
        data = f.read()
        data1 = eval(data)
        print("仪表角度：", angle, " 仪表度数：", data1[int(angle + 0.5)])
