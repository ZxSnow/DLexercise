import cv2
import numpy as np


def countIoU(recA, recB):
    xA = max(recA[0], recB[0])
    yA = max(recA[1], recB[1])
    xB = min(recA[2], recB[2])
    yB = min(recA[3], recB[3])

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    recA_area = (recA[2] - recA[0] + 1) * (recA[3] - recA[1] + 1)
    recB_area = (recB[2] - recB[0] + 1) * (recB[3] - recB[1] + 1)

    iou = inter_area / float(recA_area + recB_area - inter_area)

    return iou


img = np.zeros((512, 512, 3), np.uint8)
img.fill(255)

recA = [50, 50, 300, 300]
recB = [60, 60, 320, 320]

cv2.rectangle(img, (recA[0], recA[1]), (recA[2], recA[3]), (0, 255, 0), 5)
cv2.rectangle(img, (recB[0], recB[1]), (recB[2], recB[3]), (255, 0, 0), 5)

IoU = countIoU(recA, recB)
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, "IoU = %.2f" % IoU, (130, 190), font, 0.8, (0, 0, 0), 2)

cv2.imshow("image", img)
cv2.waitKey()
cv2.destroyAllWindows()
