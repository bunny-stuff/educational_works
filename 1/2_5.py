import numpy
import matplotlib.pyplot as ppl

fgr, grph = ppl.subplots(1, 2, figsize=(8, 4))

x = numpy.linspace(0, 5, 100)
y = [(i**3 + i**2) for i in x]

xd = x[0: -1] - x[1:]
yd = numpy.array(y[0: -1]) - numpy.array(y[1:])

xdd = (x[0:-1] + x[1:])/2
ydd = numpy.divide(yd, xd)

#grph.plot(xdd, ydd)
#grph.plot(x, (3*x**2 + 2*x))

grph[0].scatter(xdd, ydd, 2)
grph[0].set_title('По определению')
grph[1].scatter(x, (3*x**2 + 2*x), 2)
grph[1].set_title('Аналитически')

ppl.show()