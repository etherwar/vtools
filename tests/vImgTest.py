from vtools.vimg import *
import cv2

def test1():
    a = vImg(width=300, height=300, color=RED)
    b = vImg('images/car.png')
    c = b.threshold(215)
    cv2.imshow('Test1', c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test2():
    # initialize the vImg object from the quiz image
    quiz1 = vImg('images/quiz1.png')

    # take the quiz image and perform an auto canny thresholding operation,
    # then convert the result to a list of vContour objects
    cnts = quiz1.autoCanny().simpleContours()

    for i, c in enumerate(cnts, 1):
        cv2.drawContours(quiz1, [c], -1, WHITE, 1)
        cv2.putText(quiz1, f'#{i}', (c.x, c.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cvtColor(YELLOW), 2)
        print("""Shape #{i} @ x({x1},{x2}) y({y1}, {y2})
        --------------------------------------------------------------
        width: {width} height: {height}
        Aspect Ratio is (image width / image height): {aspect_ratio:.2f}
        Contour Area is: {area:.2f}
        Bounding Box Area is: {bbarea:.2f}
        Convex Hull Area is: {hull_area:.2f}
        Solidity (Contour Area / Convex Hull Area) is: {solidity:.2f} 
        Extent (Contour Area / Bounding Box Area) is: {extent:.2f}
        Center is located at: {center}""".format(i=i, x1=c.x, x2=c.x2, width=c.width, y1=c.y, y2=c.y2, height=c.height,
                                                aspect_ratio=c.aspect_ratio, area=c.area, bbarea=c.width * c.height,
                                                hull_area=c.hull_area, solidity=c.solidity, extent=c.extent,
                                                center=c.center))
        cv2.imshow('img', quiz1)
        cv2.waitKey(0)

    cv2.destroyAllWindows()