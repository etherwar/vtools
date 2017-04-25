import numpy as np
import cv2


####################################################################################################
############################################## COLORS ##############################################
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
AQUA = (0,255,255)
MAROON = (128,0,0)
FUCHSIA = (255,0,255)
OLIVE = (128,128,0)
NAVY = (0,0,128)
TEAL = (0,128,128)
PURPLE = (128,0,128)
YELLOW = (255,255,0)

class vImg(np.ndarray):
    def __new__(cls, imgFn=None, **kwargs):
        """Initiates a vImg object using np.ndarray as base class. The vImg object extends the array
        to allow for some basic image modification methods to be executed on the base class.
        Parameters:
            imgFn = string, path to image
            **kwargs (optional, only color may be used if image=None)
            height = in pixels, height of the blank image
            width = in pixels, width of the blank image
            color = in RGB tuple, the bg color for the blank image (default: black)
            height
        """
        if not imgFn:
            try:
                if kwargs.get('img', None) is not None:
                    img = kwargs['img']
                else:
                    img = vImg.blank(kwargs['height'], kwargs['width'], kwargs.get('color',(0, 0, 0)))
                obj = np.asarray(img).view(cls)
                obj.__h, obj.__w = obj.shape[:2]
                obj.__center = (obj.w // 2, obj.h // 2)
                obj.__color = kwargs.get('color',(0, 0, 0))
                return obj
            except KeyError as ke:
                str_err = "KeyError: If 'image' argument not provided, keyword arg(s) for "
                str_err += ('width and height ' if not kwargs.get('height', None) and not kwargs.get('width', None)
                            else 'height ' if not kwargs.get('height', None)
                            else 'width ')
                str_err += 'must be provided (color is optional).'
                print(str_err)
                return

        try:
            obj = np.asarray(cv2.imread(imgFn)).view(cls)
            obj.imgFn = imgFn
            obj.__h, obj.__w = obj.img.shape[:2]
        except:
            raise ValueError("Unable to open file at {}".format(imgFn))
            return

        obj.__center = (obj.__w // 2, obj.__h // 2)
        obj.__color = kwargs.get('color', (0, 0, 0))
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.__h = getattr(obj, '__h', None)
        self.__w = getattr(obj, '__w', None)
        self.__center = getattr(obj, '__center', None)
        self.__color = getattr(obj, '__color', None)
        self.__img = obj

    def __array_wrap__(self, out_arr, context=None):
        """__array_wrap__ gets called at the end of numpy ufuncs and
        other numpy functions, to allow a subclass to set the type of
        the return value and update attributes and metadata"""
        self.__h = getattr(out_arr, '__h', None)
        self.__w = getattr(out_arr, '__w', None)
        self.__center = getattr(out_arr, '__center', None)
        self.__color = getattr(out_arr, '__color', None)
        return np.ndarray.__array_wrap__(self, out_arr, context)
        # return image

    def __eq__(self, other):
        if np.array_equal(self.__img, other.img): return True

    def blank(height, width, color=(0, 0, 0)):
        """Creates a blank image
        Parameters:
            height = in pixels, height of the blank image
            width = in pixels, width of the blank image
            color = in RGB tuple, the bg color for the blank image (default: black)
            remember, in OpenCV rgb values are stored as (B,G,R)
        """
        img = np.zeros((height, width, 3), dtype = 'uint8')
        color = (color[2], color[1], color[0])
        if color != (0, 0, 0): img[:,:] = color
        return img

    @property
    def h(self):
        return self.__h

    @h.setter
    def h(self, val):
        self.__h = val

    @property
    def w(self):
        return self.__w

    @w.setter
    def w(self, val):
        self.__w = val

    @property
    def center(self):
        if not self.__center:
            self.h, self.w = self.img.shape[:2]
            self.__center = (self.w // 2, self.h // 2)
        return self.__center

    @property
    def color(self):
        return self.__color

    @color.setter
    def color(self, val):
        self.__color = val

    @property
    def img(self):
        return self.__img

    @img.setter
    def img(self, val):
        self.__img = val

    ####################################################################################################
    ########################## Reused some of Dr. Adrian Rosebrock's code and ##########################
    ########################### comments from his excellent package imutils ############################
    #########################  and book 'Practical Python and OpenCV' below.  ##########################

    def BGR2RGB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

    def BGR2RGB(self):
        return cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)

    def translate(self, x, y):
        """ function that returns translated (shifted by x and y pixels) image
        x : int, number of pixels to move the image horizontally (positive right, negative left)
        y : int, number of pixels to move the image vertically (positive down, negative up)
        """
        # Define the translation matrix and perform the translation
        M = np.float32([[1, 0, x], [0, 1, y]])
        shifted = cv2.warpAffine(self.img, M, (self.img.shape[1], self.img.shape[0]))
        # Return the translated image
        return vImg(img=shifted)
    
    def rotate(self, angle = 90, center = None, scale = 1.0):

        if not center: center = self.center
        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(self.img, M, (self.w, self.h))

        # Return the rotated image
        return vImg(img=rotated)
    
    def resize(self, width = None, height = None, inter = cv2.INTER_AREA):
        """ function that returns a resized image based on a given width, height, or both.
        Maintains the aspect ratio of the image if given only one dimension.
        width  : int, optional, width in pixels for the resized image
        height : int, optional, height in pixels for the resized image
        inter  : cv2 CONSTANT, optional, interpolation method
        Valid values for inter:
        cv2.INTER_AREA (default)
        cv2.INTER_NEAREST
        cv2.INTER_CUBIC
        cv2.INTER_LINEAR
        cv2.INTER_LANCZOS4
        """
        # initialize the dimensions of the image to be resized and grab the image size
        dim = None
        (h, w) = self.img.shape[:2]

        # if both the width and height are None, then return the original image
        if width is None and height is None:
            return vImg(img=self.img)

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(self.img, dim, interpolation=inter)

        # set the new state and img of the resized img
        return vImg(img=resized)

    def threshold(self, T, k = 5, inverse = True):
        """ We will apply binary thresholding from the current image object and return
        the second result i.e. the threshold map (extremely basic)
        T : int, threshold pixel intensity
        k : kernel size in square pixels for gaussian blur, default to 5
        inverse: bool, whether or not to return an inverse binary threashold, default YES
        """
        # The value of k must be odd so that there's a center pixel in the matrix
        if k % 2 == 0: raise ValueError(f'k must be an odd number... not {k}')

        # First, convert the color scale of the image to grayscale, then apply a gaussian blur
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(image, (k,k), 0)

        # Next, apply the threshold to the image
        thresh_bin = cv2.THRESH_BINARY if not inverse else cv2.THRESH_BINARY_INV
        thresh = cv2.threshold(gauss, T, 255, thresh_bin)[1]

        # Finally, return the result
        return vImg(img=thresh)

    def adaptiveThreshold(self, adaptive_method, neighborhood_size, k = 5, C = 3, inverse = False):
        """ We will apply adaptive thresholding to the image object and return a threshold map
        based off of a given neighborhood size.
        adaptive_method   : cv2 constant that represents which adaptive thresholding method we will
                            use (e.g. cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        neighborhood_size : int, size of the neighborhood in which to evaluate small areas of
                            pixels in order to find an optimal value of T to apply thresholding
        k                 : int, kernel size in square pixels for gaussian blur, default to 5
        C                 : int, subtracted from the mean, giving us granular control of the adaptive
                            thresholding process, default to 3
        inverse           : bool, value representing whether or not the threshold constant used will
                            be inverse or normal. Inverse is default since it's commonly used for
                            masking.
        """
        # The value of k must be odd so that there's a center pixel in the matrix
        if k % 2 == 0: raise ValueError(f'k must be an odd number... not {k}')

        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        # First, convert the color scale of the image to grayscale, then apply a gaussian blur
        image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gauss = cv2.GaussianBlur(image, (k,k), 0)

        # Next, apply the threshold to the image
        thresh_bin = cv2.THRESH_BINARY if not inverse else cv2.THRESH_BINARY_INV



def cvtColor(color):
    """Convert RGB to BGR color or vice versa"""
    return color[::-1]


if __name__ == '__main__':
    a = vImg(width=300,height=300)
    b = vImg('../../images/trex.png')
    c = b.threshold(215)
    cv2.imshow('Test1', c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
