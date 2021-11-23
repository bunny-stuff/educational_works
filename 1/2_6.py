import numpy

size = 150
V = numpy.random.random(size=(size, 2)) - 0.5
q = numpy.array([4,1])
amount_90 = 0
amount_30 = 0
for i in V:
    cos = i.dot(q)/(numpy.linalg.norm(q) * numpy.linalg.norm(i))
    if numpy.arccos(cos) * 180/numpy.pi < 90:
        amount_90 += 1
    if numpy.arccos(cos) * 180/numpy.pi < 30:
        amount_30 += 1
print('Векторов с углом меньше 90:', amount_90 * 100/size, '%')
print('Векторов с углом меньше 30:', amount_30 * 100/size, '%')