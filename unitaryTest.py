#!/usr/bin/env python3

"""
Script to expertiment and verify matrix multiplications as well
as some parts of the script
""" 

import numpy as np
import utils

def testMath():
    # Test the array multiplication
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    xyT = utils.dotxyt(x, y)
    print("x*y': ", xyT)
    print("Type of x*y': ", type(xyT))
    print("Type of x: ", type(x))
    print("Type of y': ", type(y))
    print("Shape of x*y': ", xyT.shape)
    print("Shape of x': ", x.shape)
    print("Shape of y': ", y.shape)
    
def testGradient():
    # Test gradients
    x = someValue
    epsilon = 0.0001
    numericalGradient = (f(x-epsilon) + f(x+epsilon))/(2*epsilon)
    computedGradient  = grad(x)


def main():
    testMath()
    pass


if __name__ == "__main__":
    main()
