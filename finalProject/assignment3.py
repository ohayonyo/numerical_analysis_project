"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and
the leftmost intersection points of the two functions.
The functions for the numeric answers are specified in MOODLE.
This assignment is more complicated than Assignment1 and Assignment2 because:
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors.
    2. You have the freedom to choose how to calculate the area between the two functions.
    3. The functions may intersect multiple times. Here is an example:
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately.
       You should explain why in one of the theoretical questions in MOODLE.
"""

import numpy as np
import time
import random
import assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:

        if n == 1:
            h = float(b - a)
            x0 = (a + b) / 2.0
            return np.float32(h * f(x0))
        elif n == 2:
            return np.float32(((float(b - a)) / 2.0) * (f(a) + f(b)))

        if n % 2 == 0:
            n -= 2
        else:
            n -= 1

        h = (b - a) / float(n)
        # result = f(a) + f(b)
        result_of_even = 0
        result_of_odds = 0


        for i in range(1, n, 1):
            if i % 2 == 0:
                result_of_even += f(a + i * h)
            else:
                result_of_odds += f(a + i * h)
        result = 2*result_of_even+4*result_of_odds+f(a)+f(b)
        result *= (h / 3.0)
        return np.float32(result)

    def areabetween(self, f1: callable, f2: callable) -> np.float32:

        a = 1
        b = 100
        res = 0.0
        x_arr = np.array([])
        diff = lambda x: np.abs(f1(x) - f2(x))
        # finding intersection points between the domain [a,b] which is given to us as [1,100]
        intersections_points = assignment2.Assignment2().intersections(f1, f2, a, b)
        for x in intersections_points:
            x_arr = np.append(x_arr, [x])

        # # check if there are less than 2 intersection points what gives as an "open space"
        if len(x_arr) < 2:
            return np.nan

        i = 0
        while i + 1 < len(x_arr):
            n = int((x_arr[i + 1] - x_arr[i]) * 100)
            if n == 0:
                i += 1
                continue

            res += self.integrate(diff, x_arr[i], x_arr[i + 1], n)
            i += 1
        res = np.float32(res)
        return res

        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
        Note, there is no such thing as negative area.

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.

        Parameters
        ----------
        f1,f2 : callable. These are the given functions
        Returns
        -------
        np.float32
            The area between function and the X axis
        """


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEquals(r.dtype, np.float32)

    def test_integral(self):
        ass3 = Assignment3()
        f3 = lambda x: np.sin(x * x)

        r = ass3.integrate(f3, 5, 1.5, 23)
        print("r=", r)

        # self.assertEquals(r.dtype, np.float32)

    def test_area_f1(self):
        ass3 = Assignment3()
        f1 = lambda x: 1 - 2 * x * x + x ** 3
        f2 = lambda x: x
        r = ass3.areabetween(f1, f2)
        true_result = 2.76555
        print("result of area sqrt x with x = ", r, "\t", "expect: ", true_result)
        print("difference:" + str(abs(true_result - r)))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_area_f2(self):
        ass3 = Assignment3()
        f1 = lambda x: x
        f2 = lambda x: 1
        r = ass3.areabetween(f1, f2)
        true_result = 4901
        print("result of area sqrt x with x = ", r, "\t", "expect: ", true_result)
        print("difference:" + str(abs(true_result - r)))
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
