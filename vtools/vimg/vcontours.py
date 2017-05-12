import numpy as np
import cv2
from .config import eprint


####################################################################################################
########################################## BEGIN vContour ##########################################

class vContour(np.ndarray):
    """ The vContour class is a helper class for extending contours identified by opencv with
        easily accessed properties for simple to advanced contour analysis
    """
    def __new__(cls, cnt):
        return np.asarray(cnt).view(cls)

    def __array_finalize__(self, obj):
        """ this is where we initialize most of the variables for the vContour class, due to the way
            numpy n-dimensional arrays work.
        """
        if obj is None: return
        self.__x, self.__y, self.__w, self.__h = cv2.boundingRect(obj)
        self.__x2 = self.__x + self.__w
        self.__y2 = self.__y + self.__h
        self.__aspect_ratio = self.__w / self.__h
        self.__perim = None
        self.__area = None
        self.__extent = None
        self.__hull = None
        self.__hull_area = None
        self.__solidity = None
        self.__center = None
        self.__approx = None



    def __array_wrap__(self, out_arr, context=None):
        """__array_wrap__ gets called at the end of numpy ufuncs and
        other numpy functions, to allow a subclass to set the type of
        the return value and update attributes and metadata"""
        self.__x, self.__y, self.__w, self.__h = cv2.boundingRect(out_arr)
        self.__x2 = self.__x + self.__w
        self.__y2 = self.__y + self.__h
        self.__aspect_ratio = self.__w / self.__h
        self.__perim = None
        self.__area = None
        self.__extent = None
        self.__hull = None
        self.__hull_area = None
        self.__solidity = None
        self.__center = None
        self.__approx = None
        # return image
        return np.ndarray.__array_wrap__(self, out_arr, context)

    @classmethod
    def fromList(cls, cnts):
        """ fromList is a classmethod that can be used to return a generator of vContour objects
            from a list of opencv contours returned from the cv2.findContours method"""
        if isinstance(cnts, list):
            return vContours(cls(c) for c in cnts)
        else:
            raise ValueError('fromList() constructor requires a list of contours of type nd.array') from None

    def __eq__(self, other):
        return True if np.array_equal(self, other) else False

    @property
    def x(self):
        return self.__x

    @property
    def x2(self):
        return self.__x2

    @property
    def y(self):
        return self.__y

    @property
    def y2(self):
        return self.__y2

    @property
    def w(self):
        return self.__w

    @property
    def h(self):
        return self.__h

    @property
    def width(self):
        return self.__w

    @property
    def height(self):
        return self.__h

    @property
    def center(self):
        if self.__center is None:
            # compute the moments of the contour which can be used to compute the
            # centroid or "center of mass" of the region
            M = cv2.moments(self)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            self.center = (cX, cY)
        return self.__center

    @center.setter
    def center(self, val):
        self.__center = val

    @property
    def aspect_ratio(self):
        return self.__aspect_ratio

    @property
    def perim(self):
        if self.__perim is None:
            self.perim = cv2.arcLength(self, True)
        return self.__perim

    @perim.setter
    def perim(self, val):
        self.__perim = val

    @property
    def area(self):
        if self.__area is None:
            self.area = cv2.contourArea(self)
        return self.__area

    @area.setter
    def area(self, val):
        self.__area = val

    @property
    def extent(self):
        if self.__extent is None:
            self.extent = self.area / (self.width * self.height)
        return self.__extent

    @extent.setter
    def extent(self, val):
        self.__extent = val

    @property
    def hull(self):
        if self.__hull is None:
            self.hull = cv2.convexHull(self)
        return self.__hull

    @hull.setter
    def hull(self, val):
        self.__hull = val

    @property
    def hull_area(self):
        if self.__hull_area is None:
            self.hull_area = cv2.contourArea(self.hull)
        return self.__hull_area

    @hull_area.setter
    def hull_area(self, val):
        self.__hull_area = val

    @property
    def solidity(self):
        if self.__solidity is None:
            self.solidity = self.area / self.hull_area
        return self.__solidity

    @solidity.setter
    def solidity(self, val):
        self.__solidity = val

    @property
    def approx(self):
        if self.__approx is None:
            eprint('Error: approximation not yet performed. Please make sure to run \n'
                   'getApprox(epsilon) function before attempting to access this property.\n')
        else:
            return self.__approx

    @approx.setter
    def approx(self, val):
        self.__approx = val

    def getApprox(self, epsilon = 0.01, closed = True):
        self.approx = cv2.approxPolyDP(self, epsilon * self.perim, closed)
        return self.approx


####################################################################################################
######################################### BEGIN vContours ##########################################

class vContours(list):

    def __init__(self, iterable):
        super(vContours, self).__init__(iterable)

    def copy(self):
        return vContours(self)

    def append(self, val):
        return vContours(super().append(val))

    def __copy__(self):
        return vContours(self)

    def __add__(self, rhs):
        return vContours(super().__add__(rhs))

    def __getitem__(self, item):
        return vContours(super().__getitem__(item))

    def __setitem__(self, key, val):
         return vContours(super().__setitem__(key, val))

    def __delitem__(self, item):
        return vContours(super().__getitem__(item))

    def __setslice__(self, i, j, seq):
        return vContours(super().__setitem__(slice(i, j), seq))

    def __delslice__(self, i, j):
        return vContours(super().__delitem__(slice(i, j)))

    def directionSort(self, method="left-to-right"):
        """ sorts vContours list by direction. Performs in-place sort.
        method : string, default: 'left-to-right'. Other valid values include 'right-to-left', 
                 'top-to-bottom', and 'bottom-to-top'
        """
        # handle if we need to sort in reverse
        reverse = True if method == "right-to-left" or method == "bottom-to-top" else False

        # handle sorting on y axis or x axis
        if method == "top-to-bottom" or method == "bottom-to-top":
            super(vContours, self).sort(key=lambda c: c.y, reverse=reverse)
        else:
            super(vContours, self).sort(key=lambda c: c.x, reverse=reverse)

    def sizeSort(self, reverse = True):
        """ sorts vContours list by size. Performs in-place sort. Defaults to large to small.
        reversed : bool, default: True, defaults sorts by size from large to small. If set to False,
                   will sort vContours from small to large.
        """
        super(vContours, self).sort(key=cv2.contourArea, reverse=reverse)
