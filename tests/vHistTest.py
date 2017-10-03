""" Python tests for vtools' vHist class """

import cv2
from vtools import vImg, vHist


TEST_CAR = "E://Dropbox//code//vtools root//tests//images//car.png"

# Initialize test image (car)
TEST_IMG = cv2.imread(TEST_CAR)

# Split test image into three channels
B, G, R = cv2.split(TEST_IMG)
TEST_IMG_BW = cv2.cvtColor(TEST_IMG, cv2.COLOR_BGR2GRAY)


TEST_HIST_1D = cv2.calcHist(TEST_IMG_BW, (0,), None, (256,), (0, 256))
TEST_HIST_2D = cv2.calcHist(TEST_IMG, (0, 1), None, (256, 256), (0, 256, 0, 256))
TEST_HIST_3D = cv2.calcHist(TEST_IMG, (0, 1, 2), None, (256, 256, 256), (0, 256, 0, 256, 0, 256))



car = vImg(TEST_CAR)
hists = car.histogram('1D', (True, True, True), display=True, normal=True)


# sum(1 for x in range(257) for y in range(257) for z in range(257) if TEST_HIST_3D[x][y][z] > 0)
