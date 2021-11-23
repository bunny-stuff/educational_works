def calculate(x, w, b):
    sum = 0
    if len(x) != len(w): return -1
    for n in x:
        sum += x[n] * w[n] + b
    return sum

x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
w = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
b = 30

solution = calculate(x, w, b)
print(solution)