import cv2
from conf.MeterConf import Meter5, Meter1
from detection import Detection
import time

# 检测识别框架

if __name__ == '__main__':
    # 图片剪裁
    img = cv2.imread("pic/demo-1.jpg")
    meter = Meter1
    cropped = img[meter['y0']:meter['y1'], meter['x0']:meter['x1']]
    cv2.imwrite("out/test1.jpg", cropped)

    # # 1.产生表值对应文件
    Detection(cropped, "1").scale_write()
    # # 2.进行读数
    # t1 = time.time()
    # Detection(cropped, "5").read_value()
    # t2 = time.time()
    # print("速度：", t2 - t1)
