from typing import Union, List, Tuple
from numpy import ndarray
from .vcontours import vContour, vContours
from .colors import vColor

# contour_list_type will accept either lists of type ndarray or vcontour,
# as well as vContours (which is a list subclass)
contour_list_type = Union[List[Union[List[ndarray],vContour]], vContours]

# color_type will accept either Tuples of three integers or a vColor class object
color_type = Union[Tuple[int, int, int], vColor]

# check_odd is a type pertaining to the vImg private class _isOdd that handles
# cases when the user incorrectly passes an even number for a value of k
check_odd = Union[Tuple[int,int], int]




