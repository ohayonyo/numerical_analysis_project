"""
In this assignment you should interpolate the given function.
"""
import math

import numpy as np
import time
import random


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def cubicBezierFunc(self,k0: tuple, k1: tuple, k2: tuple, k3: tuple):
        return lambda t : np.power(1 - t, 3)*k0+np.power(1 - t, 2)*3*t*k1+3*(1 - t)*np.power(t,2)*k2+np.power(t, 3)*k3

    def TomasAlgorithm(self,d_arr):
        n=len(d_arr)
        alphaArr=np.zeros(n)
        betaArr=np.zeros(n)
        vArr=np.zeros((n,2))
        xArr=np.zeros((n,2))

        alphaArr[0]=2
        betaArr[0]=0
        vArr[0]=d_arr[0]
        i=1
        while i<n-1:
            betaArr[i]=1.0/alphaArr[i-1]
            alphaArr[i]=4.0-betaArr[i]
            vArr[i]=d_arr[i]-betaArr[i]*vArr[i-1]
            i+=1
        betaArr[n-1]=2.0/alphaArr[n-2]
        alphaArr[n-1]=7.0-betaArr[n-1]
        vArr[n-1]=d_arr[n-1]-betaArr[n-1]*vArr[n-2]

        xArr[n-1]=vArr[n-1]/alphaArr[n-1]
        i=n-2
        while i>=0 :
            xArr[i]=(vArr[i]-xArr[i+1])/alphaArr[i]
            i-=1
        return xArr


    def createBezierCoef(self,points_arr):
        amount_of_equations=len(points_arr)-1
        d_array = np.zeros((amount_of_equations,2))
        i=0
        while i<amount_of_equations:
            d_array[i]=4*points_arr[i]+2*points_arr[i+1]
            i+=1
        d_array[0] = points_arr[0] + 2 * points_arr[1]
        d_array[amount_of_equations - 1] = 8 * points_arr[amount_of_equations - 1] + points_arr[amount_of_equations]
        # the calculation of d_array is correct!!!!
        A=self.TomasAlgorithm(d_array)
        B=np.zeros((amount_of_equations,2))
        i=0
        while i<amount_of_equations-1:
            B[i] = 2*points_arr[i+1]-A[i + 1]
            i+=1
        B[amount_of_equations-1] = (A[amount_of_equations - 1] + points_arr[amount_of_equations]) / 2
        return A,B


    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        if n==1 :
            return lambda x : f(a)
        if n==2 :
            y1=f(a)
            y2=f(b)
            m=(y2-y1)/(b-a)
            return np.poly1d(np.array([m,-m*a+y1]))

        original_x_arr = np.arange(a, b, (b - a) / (n-1))
        original_x_arr = np.append(original_x_arr,[b])

        original_y_arr = np.array([f(float(original_x_arr[i])) for i in range(n)])

        if type(original_x_arr) is not type(original_y_arr):
            return lambda x : original_y_arr
        points_array= np.zeros((len(original_x_arr),2))
        i=0
        while i<len(original_y_arr):
            points_array[i]=[original_x_arr[i],original_y_arr[i]]
            i+=1

        A, B = self.createBezierCoef(points_array)
        bezier_curves=[
            self.cubicBezierFunc(points_array[i], A[i], B[i], points_array[i + 1])
            for i in range(len(points_array) - 1)
        ]

        def result(x):
            if x==b :
                index=n-2
            else:
                index=math.floor((x-a)/((b-a)/(n-1)))

            if index == n-1:
                index=n-2

            t=np.abs((a+index*((b-a)/(n-1))-x)/((b-a)/(n-1)))
            return bezier_curves[index](t)[1]

        return result


        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.
        Note: It is forbidden to call f more than n times.
        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.
        Returns
        -------
        The interpolating function.
        """



##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):
    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        num_of_points = 100
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.random(2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("x^30 polynomial: " + str(T) + "[s]")
        print("x^30 polynomial: " + str(mean_err) + "[mean_err]")

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

    def test_with_sin(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = np.sin

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.uniform(low=-10, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(x): " + str(T) + "[s]")
        print("sin(x): " + str(mean_err) + "[mean_err]")

    def test_with_y_5(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: 5

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.uniform(low=-10, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("y=5: " + str(T) + "[s]")
        print("y=5: " + str(mean_err) + "[mean_err]")

    def test_with_sin_x_2(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 50
        for i in tqdm(range(100)):

            f = lambda x: np.sin(x ** 2)

            ff = ass1.interpolate(f, -1, 5, num_of_points)

            xs = np.random.uniform(low=-1, high=5, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(x^2): " + str(T) + "[s]")
        print("sin(x^2): " + str(mean_err) + "[mean_err]")

    def test_with_e_with_exponent(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.exp(-2 * (x ** 2))

            ff = ass1.interpolate(f, -2, 4, num_of_points)

            xs = np.random.uniform(low=-2, high=4, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("e^(-2x^2): " + str(T) + "[s]")
        print("e^(-2x^2): " + str(mean_err) + "[mean_err]")

    def test_with_arctan(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.arctan(x)

            ff = ass1.interpolate(f, -5, 5, num_of_points)

            xs = np.random.uniform(low=-5, high=5, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("arctan: " + str(T) + "[s]")
        print("arctan: " + str(mean_err) + "[mean_err]")

    def test_with_sinx_div_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.sin(x) / x

            ff = ass1.interpolate(f, 0.00001, 10, num_of_points)

            xs = np.random.uniform(low=0.00001, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(x)/x: " + str(T) + "[s]")
        print("sin(x)/x: " + str(mean_err) + "[mean_err]")

    def test_with_1_div_lnx(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 200
        for i in tqdm(range(100)):

            f = lambda x: 1 / np.log(x)

            ff = ass1.interpolate(f, 0.00001, 0.9999, num_of_points)

            xs = np.random.uniform(low=0.00001, high=1, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("1/ln(x): " + str(T) + "[s]")
        print("1/ln(x): " + str(mean_err) + "[mean_err]")

    def test_with_e_e_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 200
        for i in tqdm(range(100)):

            f = lambda x: np.exp(np.exp(x))

            ff = ass1.interpolate(f, -2, 2, num_of_points)

            xs = np.random.uniform(low=-2, high=2, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("e^e^x: " + str(T) + "[s]")
        print("e^e^x: " + str(mean_err) + "[mean_err]")

    def test_with_ln_ln_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.log(np.log(x))

            ff = ass1.interpolate(f, 1.00001, 30, num_of_points)

            xs = np.random.uniform(low=1.00001, high=30, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("ln(ln(x)): " + str(T) + "[s]")
        print("ln(ln(x)): " + str(mean_err) + "[mean_err]")

    def test_with_a_polynomial(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: 5 * (x ** 2) - 10 * x + 1

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("5x^2-10x+1:" + str(T) + "[s]")
        print("5x^2-10x+1: " + str(mean_err) + "[mean_err]")

    def test_with_a_exp_sin(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: 2 * (1 / (x * 2)) * np.sin(1 / x)

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("2^(1/x^2 )*sin(1/x):" + str(T) + "[s]")
        print("2^(1/x^2 )*sin(1/x): " + str(mean_err) + "[mean_err]")

    def test_with_sin_ln_x(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.sin(np.log(x))

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("sin(ln(x):" + str(T) + "[s]")
        print("sin(ln(x): " + str(mean_err) + "[mean_err]")

    def test_with_e_ln(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.log(np.exp(x))

            ff = ass1.interpolate(f, -10, 10, num_of_points)

            xs = np.random.uniform(low=-10, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("ln(e^x):" + str(T) + "[s]")
        print("ln(e^x): " + str(mean_err) + "[mean_err]")

    def test_with_ln_e(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: np.exp(np.log(x))

            ff = ass1.interpolate(f, 0.0000001, 10, num_of_points)

            xs = np.random.uniform(low=0.0000001, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("e^(lnx):" + str(T) + "[s]")
        print("e^(lnx): " + str(mean_err) + "[mean_err]")

    def test_with_poly_div_sin(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0
        num_of_points = 100
        for i in tqdm(range(100)):

            f = lambda x: (pow(2, (1 / (x ** 2)))) * (np.sin(1 / x))

            ff = ass1.interpolate(f, 3, 10, num_of_points)

            xs = np.random.uniform(low=3, high=10, size=2 * num_of_points)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / (2 * num_of_points)
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print("2^(1/(x^2))*sin(1/x): " + str(T) + "[s]")
        print("2^(1/(x^2))*sin(1/x):  " + str(mean_err) + "[mean_err]")


if __name__ == "main":
    unittest.main()