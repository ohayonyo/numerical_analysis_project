"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import math
from functools import cmp_to_key

import numpy as np
import time
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, samples):
        self.samples = samples
        pass

    def area(self):
        if self.samples is None:
            return 0
        ass5 = Assignment5()
        contour_func = self.contour
        return ass5.area(contour_func)

    def contour(self, n: int):
        if self.samples is None:
            return []
        if n > len(self.samples):
            n=len(self.samples)
        points_arr=np.zeros((n,2))
        for i in range(len(points_arr)):
            points_arr[i][0]=self.samples[i][0]
            points_arr[i][1] = self.samples[i][1]
        return points_arr

    def sample(self):
        pass

class Assignment5:
    def __init__(self):
        self.prev_sum = -555
        self.n = 200
        self.is_first_time = True
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

    def compare_points(self, point1, point2):
        if point1[1] < point2[1]:
            return 1
        elif point1[1] == point2[1] and point1[0] < point2[0]:
            return 1
        else:
            return -1

    def sort_points_cw(self, points):
        x_center = 0
        y_center = 0
        n = len(points)
        for i in range(n):
            x_center += points[i][0]
            y_center += points[i][1]

        x_center /= n
        y_center /= n

        for i in range(n):
            points[i][0] = points[i][0] - x_center
            points[i][1] = points[i][1] - y_center
        polars = self.cart2pol(points[:,0], points[:,1])
        polars=np.array(sorted(polars, key=cmp_to_key(self.compare_points)))
        points = self.pol2cart(polars)
        for i in range(n):
            points[i][0] = points[i][0] + x_center
            points[i][1] = points[i][1] + y_center

        return points


    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        points = contour(self.n)
        if len(points)<=1:
            return 0
        sum = 0
        for i in range(len(points) - 1):
            point1 = points[i]
            point2 = points[i + 1]
            base1 = point1[1]
            base2 = point2[1]
            h = np.abs(point1[0] - point2[0])
            if point2[0] < point1[0]:
                sum -= (base1 + base2) * (h / 2.0)
            else:
                sum += (base1 + base2) * (h / 2.0)

        if self.is_first_time or np.abs(np.abs(sum) - np.abs(self.prev_sum))/np.abs(sum) > maxerr:
            self.prev_sum = sum
            if self.n==10000:
                self.prev_sum = -555
                self.n = 200
                self.is_first_time = True
                return np.float32(np.abs(sum))
            elif self.n*2 <=10000:
                self.n = self.n * 2
            else:
                self.n=10000
            self.is_first_time = False
            return self.area(contour, maxerr)
        else:
            self.prev_sum = -555
            self.n = 200
            self.is_first_time = True
            return np.float32(np.abs(sum))

    def calculate_amount_of_samples(self,max_time,time_per_sample,percent_of_time_use):
        if percent_of_time_use == 0:
            return 0
        else:
            percents = percent_of_time_use/100.0
            amount_of_samples = max_time*percents
            amount_of_samples = int(amount_of_samples/time_per_sample)
            return amount_of_samples


    def sort_points(self,polar_points,sort_by=1):
        sorted_array = polar_points[np.argsort(polar_points[:, sort_by])]
        return sorted_array


    def cart2pol(self,x_points,y_points):
        res = np.zeros((len(x_points),2))
        for i in range(len(x_points)):
            r = np.sqrt(x_points[i] ** 2 + y_points[i] ** 2)
            theta = np.arctan2(y_points[i], x_points[i])
            if theta<0:
                theta += np.pi*2
            res[i][0]=r
            res[i][1]=theta
        return res

    def pol2cart(self,polars):
        res = np.zeros((len(polars),2))
        for i in range(len(polars)):
            x = polars[i][0]*np.cos(polars[i][1])
            y = polars[i][0]*np.sin(polars[i][1])
            res[i][0]=x
            res[i][1]=y
        return res

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        start = time.time()
        sample()
        time_to_sample = time.time()-start + 0.0001
        number_of_samples=self.calculate_amount_of_samples(maxtime,time_to_sample,40)
        if number_of_samples>10000:
            number_of_samples=10000

        points_array = []
        i=0
        while i < number_of_samples:
            new_point = sample()
            points_array.append(new_point)
            i+=1

        if len(points_array) == 0:
            return MyShape(None)

        new_points = np.array(points_array)
        x_array = np.array([])
        y_array = np.array([])
        for i in range(len(new_points)):
            x_point = new_points[i][0]
            y_point = new_points[i][1]
            x_array= np.append(x_array,[x_point])
            y_array = np.append(y_array,[y_point])


        sorted_points=self.sort_points_cw(new_points)
        sorted_x_array=np.zeros(len(sorted_points))
        sorted_y_array = np.zeros(len(sorted_points))
        for i in range(len(sorted_points)):
            sorted_x_array[i]=sorted_points[i][0]
            sorted_y_array[i]=sorted_points[i][1]

        sample_points = np.stack((sorted_x_array, sorted_y_array), axis=1)
        my_shape = MyShape(sample_points)
        return my_shape



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        print("the expected area is: " + str(np.pi))
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_bigger_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=2, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        print("the expected area is: " + str(4 * np.pi))
        self.assertLess(abs(a - 4 * np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_even_bigger_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=10, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        print("the expected area is: " + str(100 * np.pi))
        self.assertLess(abs(a - 100 * np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        print("the area is: " + str(a))
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_circle_area_from_contour(self):
        circ = Circle(cx=1, cy=1, radius=1, noise=0.0)
        ass5 = Assignment5()
        T = time.time()
        a_computed = ass5.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        print("The expected area is: " + str(a_true))
        print("the area found is: " + str(a_computed))
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)

    def test_circle_area_from_contour_2(self):
        circ = Circle(cx=2, cy=1, radius=2, noise=0.0)
        ass5 = Assignment5()
        T = time.time()
        a_computed = ass5.area(contour=circ.contour, maxerr=0.1)
        T = time.time() - T
        a_true = circ.area()
        print("The expected area is: " + str(a_true))
        print("the area found is: " + str(a_computed))
        self.assertLess(abs((a_true - a_computed) / a_true), 0.1)



if __name__ == "__main__":
    unittest.main()
