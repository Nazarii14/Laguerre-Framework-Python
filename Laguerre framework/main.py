import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import double


class Laguerre:
    def __init__(self, t=20, n=10, num_of_points=100, eps=0.1, beta=2, sigma=4):
        self.t = t
        self.n = n
        self.beta = beta
        self.sigma = sigma
        self.num_of_points = num_of_points
        self.eps = eps
    def __str__(self):
        return "t: {} n: {} number of points: {} eps: {} beta: {} sigma: {}".format(self.t, self.n, self.num_of_points, self.eps, self.beta, self.sigma)

    def input_data(self):
        _t = double(input("Enter t: "))
        _n = int(input("Enter n: "))
        _beta = int(input("Enter beta: "))
        _sigma = int(input("Enter sigma: "))
        _num_of_points = int(input("Enter number of points: "))
        _eps = float(input("Enter eps: "))

        self.t = _t
        self.n = _n
        self.beta = _beta
        self.sigma = _sigma
        self.num_of_points = _num_of_points
        self.eps = _eps
    def laguerre(self):
        l_0 = np.sqrt(self.sigma) * (np.exp(-self.beta * self.t / 2))
        l_1 = np.sqrt(self.sigma) * (1 - self.sigma * self.t) * (np.exp(-self.beta * self.t / 2))

        if self.n == 0:
            return l_0
        if self.n == 1:
            return l_1
        if self.n >= 2:
            l_next = (2 * 2 - 1 - self.t * self.sigma) / 2 * l_1 - (2 - 1) / 2 * l_0
            for j in range(3, self.n + 1):
                l_0 = l_1
                l_1 = l_next
                l_next = (2 * j - 1 - self.t * self.sigma) / j * l_1 - (j - 1) / j * l_0
            return l_next
    def tabulate_laguerre(self):
        steps = np.linspace(0, self.t, self.num_of_points)
        y_s = []
        for i in steps:
            self.t = i
            y_s.append(self.laguerre())
        return steps, y_s
    def experiment(self):
        self.t = 0
        while True:
            self.t += 0.0001
            res = []

            for i in range(self.n + 1):
                x = abs(self.laguerre())
                if x < self.eps:
                    res.append(x)
                    if i == self.n:
                        return self.t, res
                else:
                    break
    def integral_with_rectangles(self, f):
        alpha = self.sigma - self.beta
        self.num_of_points = 100
        steps = np.linspace(0, self.t, self.num_of_points)

        help1 = []
        for i in steps:
            self.t = i
            help1.append(f(i) * self.laguerre() * np.exp(-alpha * i))
        #res1 = sum([f(i) * self.laguerre(i, n, beta, sigma) * np.exp(-alpha * i) for i in steps]) * t / num_of_points
        res1 = sum(help1) * self.t / self.num_of_points

        self.num_of_points *= 2
        steps = np.linspace(0, self.t, self.num_of_points)

        help2 = []
        for i in steps:
            self.t = i
            help2.append(f(i) * self.laguerre() * np.exp(-alpha * i))
        res2 = sum(help2) * self.t / self.num_of_points
        #res2 = sum([f(i) * self.laguerre(i, n, beta, sigma) * np.exp(-alpha * i) for i in steps]) * t / num_of_points

        while abs(res2 - res1) >= self.eps:
            self.num_of_points *= 2
            res1 = res2

            help2 = []
            for i in steps:
                self.t = i
                help2.append(f(i) * self.laguerre() * np.exp(-alpha * i))
            res2 = sum(help2) * self.t / self.num_of_points
            #res2 = sum([f(i) * self.laguerre(i, n, beta, sigma) * np.exp(-alpha * i) for i in steps]) * t / num_of_points

        return res2
    def laguerre_transformation(self, f):
        to_return = []
        for i in range(self.n + 1):
            self.n = i
            to_return.append(self.integral_with_rectangles(f))
        return to_return
        #return [self.integral_with_rectangles(f) for i in range(self.n + 1)]
    def reverse_laguerre_transformation(self, lst):
        to_return = []
        for i in range(len(lst)):
            self.n = i
            to_return.append(lst[i] * self.laguerre())
        return sum(to_return) #sum([lst[i] * self.laguerre(t, i) for i in range(len(lst))])


    @property
    def t(self):
        return self._t
    @t.setter
    def t(self, value):
        self._t = value


    @property
    def n(self):
        return self._n
    @n.setter
    def n(self, value):
        self._n = value


    @property
    def num_of_points(self):
        return self._num_of_points
    @num_of_points.setter
    def num_of_points(self, value):
        self._num_of_points = value


    @property
    def eps(self):
        return self._eps
    @eps.setter
    def eps(self, value):
        self._eps = value


    @property
    def beta(self):
        return self._beta
    @beta.setter
    def beta(self, value):
        self._beta = value


    @property
    def sigma(self):
        return self._sigma
    @sigma.setter
    def sigma(self, value):
        self._sigma = value


def f(t):
    if 0 <= t <= 2 * np.pi:
        return np.sin(t - np.pi / 2) + 1
    elif t > 2 * np.pi:
        return 0


lag1 = Laguerre(7, 5, 100, 0.1, 2, 4)
#print(lag1.tabulate_laguerre())


lag2 = Laguerre(5, 20, 100, 0.01)
#print(lag2.laguerre_transformation(f))


lag3 = Laguerre(10, 20, 100, 0.01)
#print(lag3.reverse_laguerre_transformation(np.array([2, 5, 10, 0, -1])))