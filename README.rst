vtools
============

vimg README rev.003 2017/7/21
This library is a project that is the result of my foray into the realm of computer vision.
This project is a direct result of exploring and thinking about a highly simple and intuitive
way to create an image object, and then easily be able to perform a powerful set of
methodological analyses on that object, making routine tasks like thresholding and contouring
a simple endeavor following an object oriented approach.

I want to pay complete homage to Dr. Adrian Rosebrock in many ways for the content of this package.
His website is http://www.pyimagesearch.com/ . I've read his book and his blog posts about OpenCV
for a long time and this package is a direct result from the knowledge that I have gained while
and since doing so. This package borrows/adapts some of the work that Dr. Rosebrock has
written in his 'imutils' package located here: https://pypi.python.org/pypi/imutils

The goal of this package is to integrate these tools into an object oriented interface that
extends the np.ndarray class with methods and properties to create a simple image manipulation
and analysis interface similar to the functional interface that Dr. Rosebrock's imutils package
provides.


Dependencies
------------
OpenCV 3.0+ (required)

Python 3.6+ (required)

Mahotas (required)

scikit-image (required)

matplotlib (tested with 2+, required to visualize histograms)



Install vtools
--------------------
**From Source**

You should be able to clone this repository in to a directory (ex: vtools) and run setup.py:

    cd vtools && python setup.py install


**From PyPI**

::

    pip install vtools

Getting Started
---------------

thresholding (simple binary) an image before vtools' vImg class:

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

thresholding (simple binary) an image using vtools.vImg:

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
        (x, y, w, h) = cv2.boundingRect(c)
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