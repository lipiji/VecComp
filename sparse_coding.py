# -*- coding:utf-8 -*-
# vector completion using sparse coding
# Piji Li

from __future__ import division
import sys
import numpy as np
import math

class SC:
    def __init__(self, X):
        self.X = X;
    
    def sign(self, x):
        if x > 0:
            return 1;
        elif x == 0:
            return 0;
        else:
            return -1;

    def soft(self, x, lambd):
        return self.sign(x) * max(abs(x) - lambd, 0);


    def trainMP(self, b=None):
        e = 0.0001;
        iter = 300;
        eta = 1.0;
        lambd = 0.01;

        X = self.X;
        [num_x, num_f] = X.shape;

        if b == None:
            b = np.ones(num_x, dtype = np.float);
        
        a = np.zeros(num_x, dtype = np.float);
        for i in range(0, num_x):
            norm = np.linalg.norm(X[i,]);
            if norm > 0:
                X[i,:] /= norm;
        
        a_new = a.copy();

        for i in range(0, iter):
            sum_x = np.zeros((1, num_f), dtype = np.float);
            for j in range(0, num_x):
                sum_x += a[j] * X[j,];

            minJ = 0 
            for j in range(0, num_x):
                minJ += 1 / 2 /num_x * b[j] * math.pow(np.linalg.norm(X[j,] - sum_x), 2) + lambd / num_x *a[j];
            
            re_sum = np.repeat(sum_x, num_x, axis=0);
            
            grad = np.zeros(num_x, dtype = np.float);
            for k in range(0, num_x):
                grad[k] = -1 * (b * ((X - re_sum) * X[k,].T)) / num_x ;
            
            max_k = np.argmax(np.abs(grad))
            a_new[max_k] = self.soft(a[max_k] - eta * grad[max_k], lambd);
            
            error = np.linalg.norm(a_new - a);
            print i, minJ, error;
            if error < e:
                break;
            else:
                a = a_new.copy();
        return a;

def test():
    data = np.matrix([  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype = np.float)
    
    sc = SC(data);
    a = sc.trainMP();
    print a

if __name__ == "__main__":

    test();

