import numpy as np
from typing import Any
import matplotlib.pyplot as plt

class Polygon: 
    def __init__(self, vertices: np.ndarray[Any, np.dtype[np.float64]]):
        self.vertices: np.ndarray[Any, np.dtype[np.float64]] = vertices

    def Vertices(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self.vertices
    
    def Edges(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        num_vertices = len(self.vertices)
        # Crear un array tridimensional para los segmentos
        edges = np.zeros((num_vertices, 2, 2), dtype=np.float64)

        for i in range(num_vertices):
            v1 = self.vertices[i]
            v2 = self.vertices[(i + 1) % num_vertices]  # Conexión cíclica
            edges[i] = np.array([v1, v2])

        return edges
    
    def Perimenter(self) -> float:
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2) + np.sqrt((x[-1]-x[0])**2 + (y[-1]-y[0])**2))
    
    def Area(self) -> float:
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    
    def Centroid(self) -> np.ndarray[np.dtype[np.float64]]:
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        A = self.Area()
        Cx = np.sum((x + np.roll(x,1))*(x*np.roll(y,1) - y*np.roll(x,1)))/(6*A)
        Cy = np.sum((y + np.roll(y,1))*(x*np.roll(y,1) - y*np.roll(x,1)))/(6*A)
        return np.array([Cx, Cy])
    
    def PointIsInside(self, point: np.ndarray[np.dtype[np.float64]]) -> bool:
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        n = len(x)
        c = False
        for i in range(n):
            if (((y[i] <= point[1] and point[1] < y[i-1]) or (y[i-1] <= point[1] and point[1] < y[i])) and (point[0] < (x[i-1] - x[i]) * (point[1] - y[i]) / (y[i-1] - y[i]) + x[i])):
                c = not c
        return c
    
    def GetRandomPointInside(self) -> np.ndarray[np.dtype[np.float64]]:
        x = self.vertices[:,0]
        y = self.vertices[:,1]
        while True:
            point = np.array([np.random.uniform(np.min(x), np.max(x)), np.random.uniform(np.min(y), np.max(y))])
            if self.PointIsInside(point):
                return point

    def Plot(self, color='blue', linewidth=1.5, *, linestyle = 'solid', legend = None, fillColor=None):
        x = np.append(self.vertices[:,0], self.vertices[0,0])
        y = np.append(self.vertices[:,1], self.vertices[0,1])
        
        if fillColor is not None:
            lengedFill = legend
            legend = None
        plt.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle, label=legend)
        
        if fillColor is not None:
            self.FillArea(color=fillColor, legend = lengedFill)
        
    def FillArea(self, color='blue', *,legend=None):
        x = np.append(self.vertices[:,0], self.vertices[0,0])
        y = np.append(self.vertices[:,1], self.vertices[0,1])
        plt.fill(x, y, color=color)
        if legend:  # Si se proporciona un elemento de leyenda
            plt.gca().add_patch(plt.Polygon(list(zip(x, y)), color=color, label=legend))
            plt.legend()


            
    


