import numpy as numpy
import matplotlib.pyplot as ppl

rslts = []
size = 500
fgr, grph = ppl.subplots()
for i in range(1, 10):
    pts = numpy.random.random(size=(size, i))
    count = 0
    for j in pts:
        if sum([k**2 for k in j]) <= 1:
            count += 1
    rslts.append(count/size)

grph.set_xlim(2,10)
grph.set_ylim(0,1)
grph.plot(range(1, 10), rslts)

ppl.show()