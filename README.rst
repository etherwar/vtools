vtools
============

vimg README rev.003 2017/7/21
This library is a project that is the result of my foray into the realm of computer vision.
This project is a direct result of exploring and thinking about a highly simple and intuitive
way to create an image object, and then easily be able to perform a powerful set of
methodological analyses on that object, making routine tasks like thresholding and contouring
a simple endeavor following an object oriented approach.


I want to pay all due homage to Dr. Adrian Rosebrock in many ways for the content of this package.
His website is http://www.pyimagesearch.com/ . I've read his book and his blog posts about OpenCV
for a long time (and am now enrolled and working through his PyImageSearch Gurus course) and this
package is a direct result from the knowledge that I have gained while and since doing so. This package
borrows/adapts some of the work that Dr. Rosebrock has written in his open source 'imutils' package
located here: https://pypi.python.org/pypi/imutils


The vImg class (Visual Tools Image) is designed as a subclass of numpy's ndarray type that extends
ndarray to include operations that computer vision (CV) researchers and practitioners use frequently to
analyze images and procure valuable data from. In order to accomplish this, we lean heavily on computer
vision libraries that are already in place and usually optimized with code written in C to maximize
performance. This class, therefore, serves to turn images (which I would argue lend themselves inherently
to an object-oriented approach) into objects, from which methods may be called individually or chained in
a single statement in order to rapidly prototype ideas and serve as an efficient medium that is able
to explore challenging conceptual image analysis operations in a simple manner.


When writing this class, I've opted to approach this goal with simplicity of use at the forefront, so you
will likely see some areas where efficiency could be improved. That being said, I also wanted to maintain
the ability to fine-tune parameters and dial in accuracy, so that option remains available (usually through
parameter and keyword tuning). Efficiency has not been cast asunder either; any means that I have had
to optimize I have attempted to implement. I know there are opportunities for improvement, and I am very
open to suggestion as well as any potential collaborators.


I have done my best to maintain this hierarchy throughout the codebase and provide a well-documented tool
that will hopefully one day be used by more than just myself. For the time being though, I am treating this
endeavor as an exercise both in creating a package (this is my first), and to create a



Dependencies
------------
OpenCV 3.0+ (required)

Python 3.6+ (required)

Mahotas (required)

scikit-image (required)

matplotlib (required, tested with 2+)



Install vtools
--------------------
**From Source**

You should be able to clone this repository in to a directory (ex: vtools) and run setup.py:

    cd vtools && python setup.py install


**From PyPI**

    pip install vtools

Getting Started
---------------

Thresholding (simple binary) an image before vtools' vImg class:

    # Read in the image

    image = cv2.imread('../images/trex.png')

    # Convert to grayscale and apply gaussian blur
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Set gaussian blur k (size of weighted mean area),

    # must be odd so there's a center pixel
    
    k = 3

    gauss = cv2.GaussianBlur(gray, (k,k), 0)

    # Now set the threshold level, T
    
    T = 215

    # Next, apply the threshold to the image
    
    thresh = cv2.threshold(gauss, T, 255, cv2.THRESH_BINARY_INV)[1]

Thresholding (simple binary) an image using vtools.vImg:

    image = vImg('../images/trex.png')

    thresh = image.threshold(215)

note: currently the only required variable is for T, but k (defaults to 5) and
inverse (bool, defaults to True) are also available as named parameters.

The vContour class:

calculating contours and evaluating contour properties before vtools.vimg:

    image = cv2.imread('quiz1.png')

    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hullImage = np.zeros(gray.shape[:2], dtype="uint8")

    # loop over the contours
    
    for (i, c) in enumerate(cnts):
        
        # compute the area of the contour along with the bounding box

        # to compute the aspect ratio

        print(f'Contour {i} type({type(c)})')

        area = cv2.contourArea(c)

        x, y, w, h = cv2.boundingRect(c)

        x2, y2 = x + w, y + h


        # compute the aspect ratio of the contour, which is simply the width

        # divided by the height of the bounding box
        
        aspectRatio = w / float(h)


        # use the area of the contour and the bounding box area to compute

        # the extent
        
        extent = area / float(w * h)


        # compute the convex hull of the contour, then use the area of the

        # original contour and the area of the convex hull to compute the

        # solidity
        
        hull = cv2.convexHull(c)

        hullArea = cv2.contourArea(hull)

        solidity = area / float(hullArea)


        # compute the center (tuple)
        
        center = ((x + x2) / 2, (self. + y2) / 2)


        # visualize the original contours and the convex hull and initialize

        # the name of the shape
        
        cv2.drawContours(hullImage, [hull], -1, 255, -1)

        cv2.drawContours(image, [c], -1, (240, 0, 159), 3)

        print(f'Shape #{i}: Aspect Ratio is {aspectRatio:.2f}, hull area is {hullArea:.2f}, '
        f'solidity is {solidity:.2f}, extent is {extent:.2f}, center is {center}')


Evaluating contours for usefulness with vtools' vImg, vContour, and vContours classes:

    img = vImg("images/test.png")

    # outline each contour one by one and print simple and advanced contour properties

    # allowing you to easily determine whether contours may be useful to your CV application
    
    img.gray().evalContours()

    # the evalContours() method defaults to using the vImg simpleContours function with default parameters,

    # but you can also supply your own calculated contour values (in the form of a list of vContours)


Histograms with vtools' vImg

*** Coming Soon! ***