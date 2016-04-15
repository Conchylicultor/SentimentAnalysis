"""
Script to expertiment and verify matrix multiplications as well
as some parts of the script
""" 

def testGradient():
    # Test gradients
    x = someValue
    epsilon = 0.0001
    numericalGradient = (f(x-epsilon) + f(x+epsilon))/(2*epsilon)
    computedGradient  = grad(x)


def main():
    pass


if __name__ == "__main__":
    main()
