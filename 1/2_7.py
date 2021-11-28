import numpy as np
import matplotlib.pyplot as plt

results = []
size = 500
r = range(1, 30)
for i in r:
    count = 0
    points = np.random.random(size=(size, i))
    for p in points:
        if 1 >= sum([x**2 for x in p]):
            count += 1
    results.append(count/size)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_ylim(0, 0.85)
ax.set_xlim(r[0], len(r))
ax.plot(r, results, 100)

plt.show()