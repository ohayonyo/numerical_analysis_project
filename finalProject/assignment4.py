"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""
import math

import numpy as np
import time
import random
from scipy import linalg

import commons
from numpy.linalg import matrix_rank

class Assignment4A:
    def __init__(self):
        self.n = 100
        self.to_rand=False
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """



    def Lu_decomposition(self,A):
        n = len(A)
        L = np.zeros_like(A)
        U = np.zeros_like(A)

        for k in range(n):
            U[k][k]=A[k][k]
            for i in range(k,n,1):
                if U[k][k]!=0:
                    L[i][k]=A[i][k]/U[k][k]
                U[k][i]=A[k][i]
            for i in range(k+1,n,1):
                for j in range(k+1,n,1):
                    A[i][j]=A[i][j]-L[i][k]*U[k][j]

        return L,U

    def solve(self,matrix,vector_res):
        m = np.copy(matrix)
        L,U = self.Lu_decomposition(m)
        n = len(vector_res)
        c_vector = np.zeros(n)
        x_vector = np.zeros(n)


        #solving for c_vector
        for i in range(n):
            sum = 0
            for j in range(n):
                if i!=j:
                    sum+=L[i][j]*c_vector[j]
            ci = vector_res[i]-sum
            c_vector[i]=ci

        i=n-1
        while i>=0:
            sum=0
            for j in range(n):
                if i!=j:
                    sum+=U[i][j]*x_vector[j]
            xi = c_vector[i]-sum
            if U[i][i]!=0:
                xi = xi/U[i][i]
            x_vector[i]=xi
            i-=1

        return x_vector

    def matrix_has_a_solution(self,A,vector_res):
        rank_A = matrix_rank(A)
        n=len(A)
        new_matrix = np.zeros((n,n+1))
        for i in range(n):
            for j in range(n):
                new_matrix[i][j]=A[i][j]

        for i in range(n):
            new_matrix[i][n]=vector_res[i]

        rank_new_mat = matrix_rank(new_matrix)
        if rank_A == rank_new_mat:
            return True
        else:
            return False


    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:


        initial_time = time.time()
        if self.to_rand:
            x_arr = np.random.uniform(a, b, self.n)
        else:
            x_arr = np.arange(a,b,(b-a)/(self.n))
        x_arr = np.append(x_arr,[b])
        y_arr=np.zeros(len(x_arr))
        for i in range(len(y_arr)):
            y_arr[i]=f(x_arr[i])


        sum_x_pow_i = np.zeros(2*d+1)
        sum_y_mul_x_pow_i = np.zeros(d+1)
        sum_x_pow_i = np.array([np.sum(np.power(x_arr,i)) for i in range(2*d+1)])

        for i in range(d, -1, -1):
            sum_y_mul_x_pow_i[d-i] = np.sum([y_arr[j] * np.power(x_arr[j], i) for j in range(len(x_arr))])

        matrix_coeff = np.array(np.zeros((d+1,d+1)))
        for i in range(d+1):
            maxpow = d + (d-i)
            for j in range(d+1):
                matrix_coeff[i][j]=sum_x_pow_i[maxpow-j]

        vectorRes = sum_y_mul_x_pow_i
        # solve the matrix by my implementation (using Lu_decomposition)
        if not self.matrix_has_a_solution(matrix_coeff,vectorRes):
            end_time = time.time()
            running_time = end_time - initial_time
            self.to_rand=True
            time_left = maxtime - running_time
            if time_left>0.25:
                return self.fit(f,a,b,d,time_left)
            else:
                return lambda x : x

        res = self.solve(matrix_coeff, vectorRes)

        def myfunc(x):
            sum=0
            for i in range(len(res)):
                sum+=res[i]*np.power(x,d-i)
            return np.float32(sum)

        end_time = time.time()
        running_time = end_time - initial_time
        time_per_sample = running_time / self.n
        time_left = maxtime - running_time
        # print("n=",self.n)
        if 1.75*((self.n * 2) * time_per_sample) < time_left:
            self.n = (self.n * 2)
            return self.fit(f, a, b, d, time_left)
        # elif 1.2*(self.n+100)*time_per_sample < time_left:
        #     self.n = (self.n+100)
        #     return self.fit(f, a, b, d, time_left)
        else:
            self.n = 100
            return myfunc


##########################################################################

import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(7)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4A()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4A()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()