from Classes.Shape import *
from CSA import *
import numpy as np
import matplotlib.pyplot as plt 


outerPolygon = Polygon(np.array([
    [0, 0], [10, 0], [5, 4], [10, 4], [8, 6], [10, 8], [9, 10],
    [6, 9], [4, 10], [2, 8], [0, 9], [1, 6], [0, 4], [2, 2]
]))

# Primer polígono pequeño (1.5x1.5 aprox, dentro del grande)
innerPolygon1 = Polygon(np.array([
    [3, 3], [4, 3.2], [4.2, 4], [3.8, 4.5], [3, 4.2], [2.8, 3.5]
]))

# Segundo polígono pequeño (1.5x1.5 aprox, dentro del grande)
innerPolygon2 = Polygon(np.array([
    [6, 5], [7, 5.2], [7.5, 6], [7.2, 6.8], [6.5, 7], [5.8, 6.5], [6, 5.8]
]))



shape = Shape(outerPolygon, [innerPolygon1, innerPolygon2])
shape.Plot()


area_percentage = [10, 15, 25, 50]
shape.Plot(fillColor='black')
plt.gca().set_aspect('equal')
plt.show()

shape.Plot()
csa = CSA(shape, area_percentage)
csa.Plot()
plt.gca().set_aspect('equal')
plt.show()

csa.Resolve()


