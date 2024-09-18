from math import factorial


def func1(n):
    result = 0
    for k in range(n // 2 + 1):
        numerator =  (n - 2*k + 1) * factorial(n)
        denominator = factorial(n-k+1)  * factorial(k)
        result += (numerator // denominator)**2

    return result

def func2(n):
    return factorial(2*n) // (factorial(n+1) * factorial(n))


for i in range(10):
    print(func1(i), func2(i))
        
