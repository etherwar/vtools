####################################################################################################
########################################### vtools.vimg ############################################
############################################ config.py #############################################
####################################################################################################
import sys

""" Here is where I have placed all the necessary helper functions """
def _idGen():
    num = 0
    while True:
        num += 1
        yield str(num)

__IDENT__ = _idGen()

def cvtColor(color):
    """Convert RGB tuple to BGR color tuple or vice versa"""
    return color[::-1]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    