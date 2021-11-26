import numpy


x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
w = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
b = 30

solution = numpy.dot(x, w) + b
print(solution)