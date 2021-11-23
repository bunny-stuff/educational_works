import numpy
import matplotlib.pyplot as ppl

def transformation_plot(pts, mtrx):
    fgr, im = ppl.subplots(1, 2, figsize = (8, 4))
    x_dot = pts[:, 0]
    y_dot = pts[:, 1]
    im[0].scatter(x_dot, y_dot, 20, x_dot + y_dot)
    
    d = numpy.dot(pts, mtrx)
    x_dot = d[:, 0] 
    y_dot = d[:, 1]
    im[1].scatter(x_dot, y_dot, 20, x_dot + y_dot)

transformation_plot(numpy.random.random(size = (122, 2)), numpy.array([[1, 2], [3, 4]]))
ppl.show()