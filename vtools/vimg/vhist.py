import numpy as np
import cv2

####################################################################################################
######################################### Histogram Class ##########################################

class vHist(np.ndarray):

    def __new__(cls, images, channels = None, mask = None, histSize = None, ranges = None):

        """
        vHist is a class to simplify the display of different kind of histograms. Meant to be used in conjunction
        with the vImg class.
        
        images   : This is the image that we want to compute a histogram for. Wrap it as a list: [myImage] . 
        channels : A list of indexes, where we specify the index of the channel we want to compute a histogram for. 
                   To compute a histogram of a grayscale image, the list would be [0] . To compute a histogram for all 
                   three red, green, and blue channels, the channels list would be [0, 1, 2] . 
        mask     : Remember learning about masks in Section 1.4.8? Well, here we can supply a mask. If a mask is 
                   provided, a histogram will be computed for masked pixels only. If we do not have a mask or do not 
                   want to apply one, we can just provide a value of None . 
        histSize : This is the number of bins we want to use when computing a histogram. Again, this is a list, one 
                   for each channel we are computing a histogram for. The bin sizes do not all have to be the same. 
                   Here is an example of 32 bins for each channel: [32, 32, 32] . 
        ranges   : The range of possible pixel values. Normally, this is [0, 256] (this is not a typo — the ending 
                   range of the cv2.calcHist  function is non-inclusive so you’ll want to provide a value of 256 rather 
                   than 255) for each channel, but if you are using a color space other than RGB [such as HSV], 
                   the ranges might be different.)
        """
        assert hasattr(images, '__iter__'), "images must be a list"

        assert not(channels is None), "None check failed for channels"
        assert hasattr(channels, '__iter__'), "channels must be a list"
        if not isinstance(channels, list):
            channels = list(channels)

        assert not(histSize is None), "None check failed for histSize"
        assert hasattr(histSize, '__iter__'), "histSize must be a list"
        if not isinstance(histSize, list):
            histSize = list(histSize)

        assert not(ranges is None), "None check failed for ranges"
        assert hasattr(ranges, '__iter__'), "ranges must be a list"
        if not isinstance(ranges, list):
            ranges = list(ranges)

        obj = cv2.calcHist(images, channels, mask, histSize, ranges)

        obj = np.asarray(obj).view(cls)

        if len(obj.shape) == 2:
            obj.__type = '2D'

        obj.__images = images
        obj.__channels = channels
        obj.__mask = mask
        obj.__histSize = histSize
        obj.__ranges = ranges

        return obj

    def __array_finalize__(self, obj):
        """ this is where we initialize most of the variables for the vHist class, due to the way
            numpy n-dimensional arrays work.
        """
        if obj is None: return
        self.__images = getattr(obj, '__images')
        self.__channels = getattr(obj, '__channels')
        self.__mask = getattr(obj, '__mask')
        self.__histSize = getattr(obj, '__histSize')
        self.__ranges = getattr(obj, '__ranges')

    def __array_wrap__(self, out_arr, context=None):
        """__array_wrap__ gets called at the end of numpy ufuncs and
        other numpy functions, to allow a subclass to set the type of
        the return value and update attributes and metadata"""
        self.__images = getattr(out_arr, '__images')
        self.__channels = getattr(out_arr, '__channels')
        self.__mask = getattr(out_arr, '__mask')
        self.__histSize = getattr(out_arr, '__histSize')
        self.__ranges = getattr(out_arr, '__ranges')
        # return image
        out_arr.__images = getattr(self, '__images')
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __eq__(self, other):
        return True if np.array_equal(self, other) else False

    #TODO: Consider if a single show function would be best implementation.
    """
    PROS:
    
    
    CONS:
    
    """

    def showFlat(self, RGB = (True, True, True)):
        """" showFlat does what you might expect, it displays a flat histogram using matplotlib's pyplot
        """
        #TODO: Finish documentation

        assert len(RGB) == 3, "If specifying channels with RGB, " \
            "must specify with 3-tuple with True or False " \
            "representing RGB color channel inclusion."



    def show2D(self, RGB = (False, False, False)):
        assert len(RGB) == 3 and sum(1 for e in RGB if e == True), \
            "Must provide show2d a 3-tuple with True or False " \
            "representing RGB color channel inclusion."
