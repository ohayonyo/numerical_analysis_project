"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def regulaFalsi(self, f1: callable, f2: callable, x1: float, x2: float, a: float, b: float, maxerr=0.001) -> float:
        max_amount_of_iteration_loop = 30
        f_x1 = f1(x1) - f2(x1)
        f_x2 = f1(x2) - f2(x2)
        if f_x1 == f_x2:
            return None
        x = (x1 * f_x2 - x1 * f_x1) / (f_x2 - f_x1)

        i = 0
        while np.abs(f1(x) - f2(x)) >= maxerr and a - maxerr <= x <= b + maxerr and i < max_amount_of_iteration_loop:

            x = (x1 * f_x2 - x2 * f_x1) / (f_x2 - f_x1)

            if np.abs(f1(x) - f2(x)) <= maxerr:
                break
            elif f_x1 * (f1(x) - f2(x)) < 0:
                x1 = x
                f_x1 = f1(x1) - f2(x1)
            else:
                x2 = x
                f_x2 = f1(x2) - f2(x2)
            i += 1
        if i > max_amount_of_iteration_loop or (x > b or x < a) or np.abs(f1(x) - f2(x)) > maxerr:
            return None
        return x

    def helper(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        diff = b - a
        int_of_diff = int(diff)

        max_amount_of_points = int_of_diff * 50
        x1 = a

        # check this if
        if max_amount_of_points == 0:
            max_amount_of_points = 50

        delta = (b - a) / max_amount_of_points
        x2 = x1
        f_x1 = f1(x1) - f2(x1)
        f_x2 = f1(x2) - f2(x2)
        while x1 <= b + maxerr and x2 <= b + maxerr:

            if abs(f_x1) <= maxerr:
                yield x1
                x1 += delta
                x2 = x1 + delta
                f_x1 = f1(x1) - f2(x1)
            elif abs(f_x2) <= maxerr:
                yield x2
                x1 = x2 + delta
                x2 = x1 + delta
                # x2 = x2 + delta
                # x1 = x2 + delta
                f_x1 = f1(x1) - f2(x1)
            elif f_x1 * f_x2 < 0 and f_x1 != f_x2:
                x = self.regulaFalsi(f1, f2, x1, x2, a, b, maxerr)
                # print("here x="+str(x))
                # print("f1(x)-f2(x)="+str(f1(x))+"-"+str(f2(x))+"="+str(f1(x)-f2(x)))
                if x is not None:
                    yield x
                x1 = x2 + delta
                x2 = x2 + delta
                f_x1 = f1(x1) - f2(x1)
            else:
                x1 = x2
                x2 += delta

            f_x2 = f1(x2) - f2(x2)

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        iterator = self.helper(f1, f2, a, b, maxerr)
        arr = np.array([])
        for x in iterator:
            if len(arr) == 0 or abs(x - arr[len(arr) - 1]) > maxerr:
                arr = np.append(arr, [x])
        return iter(arr)

        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # replace this line with your solution


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):
    f1 = 5
    f2 = np.poly1d(np.array([1, -3, 5]))
    f3 = lambda x: np.sin(x * x)
    f4 = lambda x: np.e ** (-2 * x * x)
    f5 = lambda x: np.arctan(x)
    f6 = lambda x: np.sin(x) / x

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print("testing sqrt:")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
            print("x=" + str(x))

    def test_sqr2(self):

        ass2 = Assignment2()

        f1 = lambda x: -x * x
        f2 = lambda x: x * x

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        print("testing sqrt2:")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
            print("x=" + str(x))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
        # print("testing poly:")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
        # print("x=" + str(x))

    def test_poly2(self):

        ass2 = Assignment2()

        f1 = np.array([2, 3, 0, 0, -4, 0])
        f2 = np.array([1, -2, 0, 5])
        f1 = np.poly1d(f1)
        f2 = np.poly1d(f2)

        X = ass2.intersections(f1, f2, -2.5, 2.5, maxerr=0.001)
        print("testing poly2:")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
            print("x=" + str(x))

    def test_sin(self):

        ass2 = Assignment2()

        f1 = lambda x: np.sin(x * x)
        f2 = lambda x: np.e ** (-2 * x * x)
        X = ass2.intersections(f1, f2, -2.6, 2.6, maxerr=0.001)
        print("testing sin:")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
            print("x=" + str(x))

    def test_sin2(self):

        ass2 = Assignment2()

        f1 = lambda x: np.sin(1 / x)
        f2 = lambda x: np.e ** (-2 * x * x)
        X = ass2.intersections(f1, f2, -2.6, 2.6, maxerr=0.001)
        print("testing sin2:")
        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
            print("x=" + str(x))


if __name__ == "__main__":
    unittest.main()
