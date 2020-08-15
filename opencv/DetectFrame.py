import time

from conf.MeterConf import Meter5
from opencv.detection import Detection

# 检测识别框架

if __name__ == '__main__':
    # img = cv2.imread("pic/demo-1.jpg")
    # cropped = img[meter['y0']:meter['y1'], meter['x0']:meter['x1']]
    # cv2.imwrite("out/test1.jpg", cropped)

    # # 1.产生表值对应文件
    read = Detection(path='../pic/demo-5.jpg')
    read.meter = Meter5
    read.cut()
    # read.scale_write()
    # 2.进行读数
    t1 = time.time()
    read.read_value()
    t2 = time.time()
    print("速度：", t2 - t1)
