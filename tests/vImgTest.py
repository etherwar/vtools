from vtools.vimg import *
import cv2

a = vImg(width=300, height=300)
b = vImg('../../images/trex.png')
c = b.threshold(215)
cv2.imshow('Test1', c)
cv2.waitKey(0)
cv2.destroyAllWindows()