import numpy as np
from Classes.Polygon import * 

HEXAGON_ORIENTATION = 'FLAT'  # 'FLAT' /--\    or 'POINTY' /\ Top form of the hexagon 
class Hexagon: 
    def __init__(self, center: np.ndarray[np.dtype[np.float64]], radius: float): 
        self.center = center #polygon centroid
        self.radius = radius #radius == edge length

        vertices = None
        if HEXAGON_ORIENTATION == 'FLAT':
            vertices = np.array([[self.center[0] + self.radius*np.cos(np.pi/3*i), self.center[1] + self.radius*np.sin(np.pi/3*i)] for i in range(6)])
        else:
            vertices = np.array([[self.center[0] + self.radius*np.cos(np.pi/6 + np.pi/3*i), self.center[1] + self.radius*np.sin(np.pi/6 + np.pi/3*i)] for i in range(6)])
        self.polygon = Polygon(vertices)
        
def Neighbors() -> np.ndarray[Any ,np.dtype[np.int32]]: 
    if HEXAGON_ORIENTATION == 'FLAT':
        neighbors_deltas = np.array([[1, 1], [2, 0], [1, -1], [-1, -1], [-2, 0], [-1, 1]])
    else:
        neighbors_deltas = np.array([[0, 2], [1, 1], [1, -1], [0, -2], [-1, -1], [-1, 1]])
    return neighbors_deltas

def SwapOrientation(swapPosition: np.ndarray[Any ,np.dtype[np.int32]] ) -> np.ndarray[Any ,np.dtype[np.int32]]:
    j, i = swapPosition
    if j % 2 == 0:
        connection = 0
    else:
        if ((i-1)%4 == 0 and (j-1)%4 == 0) or ((i-3)%4 == 0 and (j-3)%4 == 0):
            connection = 1
        else:
            connection = 2
    if HEXAGON_ORIENTATION == 'FLAT':
        diagonals = [(2, 0), (1, 1), (1, -1)]
    elif HEXAGON_ORIENTATION == 'POINTY':
        diagonals = [(0, 2), (1, 1), (1, -1)]
    
    dj, di = diagonals[connection]
    return  np.array([[(j-dj)//2, (i-di)//2], [(j+dj)//2, (i+di)//2]])

