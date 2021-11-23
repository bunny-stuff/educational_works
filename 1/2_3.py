import matplotlib.pyplot as plt
import numpy as np

def draw_circle(r):
    circle = plt.Circle((0, 0), r, fill=False)
    ax=plt.gca()
    ax.add_patch(circle)
    plt.axis('scaled')
    plt.show()

r = int(input('r:'))
print(r)
draw_circle(r)