vtools
============

vimg README rev.0 2017/3/11
This library is a project that is the result of my venture into the realm of computer vision.
This project is a direct result of exploring and thinking about a highly simple and intuitive
way to create an image object, and then easily be able to perform a powerful set of methods and
alterations to that object, making routine tasks like thresholding and contouring a more simple
and Object-oriented endeavor.

I want to pay complete homage to Dr. Adrian Rosebrock in every way for the content of this package.
His website is http://www.pyimagesearch.com/ . I've read his book and his blog posts about OpenCV
for a long time and this package is a direct result from the knowledge that I have gained while
and since doing so. This package borrows logic, code, and even comments that Dr. Rosebrock has
written in his 'imutils' package located here: https://pypi.python.org/pypi/imutils

The goal of this package is to integrate these tools into an object oriented interface that extends
the np.ndarray class with methods and properties to create an even simpler image manipulation and
analysis tool that what Dr. Rosebrock's imutils package provides.


Dependencies
------------
OpenCV 3.0+
Python 3.6+


Install vtools
--------------------
**From Source**

You should clone this repository and run setup.py::

    cd vtools && python setup.py install

**From PyPI**

::

    pip install vtools

Getting Started
---------------

FURTHER INSTRUCTIONS INCOMING AT LATER DATE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~