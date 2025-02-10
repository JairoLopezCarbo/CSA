from Classes.Hexagon import *
from Classes.Polygon import *
from typing import List, Tuple

class Shape:
    def __init__(self, outerPolygon: Polygon, innerPolygons = List[Polygon]): 
        self.outerPolygon: Polygon = outerPolygon  
        self.innerPolygons: List[Polygon] = innerPolygons
        
    def Vertices(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        outerVertices = self.outerPolygon.Vertices()
        innerVertices = []
        for hole in self.innerPolygons:
            innerVertices.append(hole.Vertices())
        return outerVertices
    
    def Edges(self) -> np.ndarray[np.dtype[np.float64]]: 
        outerEdges = self.outerPolygon.Edges()
        innerEdges = []
        for hole in self.innerPolygons:
            innerEdges.append(hole.Edges())
        return outerEdges, innerEdges
    
    def Perimeter(self) -> float:
        P = self.outerPolygon.Perimeter()
        for hole in self.innerPolygons:
            P += hole.Perimeter()
        return P
    
    def Area(self)  -> float:
        A = self.outerPolygon.Area()
        for hole in self.innerPolygons:
            A -= hole.Area()
        return A
    
    def Centroid(self) -> np.ndarray[np.dtype[np.float64]]:
        A = self.Area()
        Cx = self.outerPolygon.Area()*self.outerPolygon.Centroid()[0]
        Cy = self.outerPolygon.Area()*self.outerPolygon.Centroid()[1]
        for hole in self.innerPolygons:
            Cx -= hole.Area()*hole.Centroid()[0]
            Cy -= hole.Area()*hole.Centroid()[1]
        return np.array([Cx/A, Cy/A])
    
    def PointIsInside(self, point: np.ndarray[np.dtype[np.float64]]) -> bool:
        if not(self.outerPolygon.PointIsInside(point)):
            return False
        for hole in self.innerPolygons:
            if hole.PointIsInside(point):
                return False
        return True
    
    def GetRandomInsidePoint(self) -> np.ndarray[np.dtype[np.float64]]:
        while True:
            point = self.outerPolygon.GetRandomInsidePoint()
            for hole in self.innerPolygons:
                if hole.PointIsInside(point):
                    continue
            return point
        
    def Plot(self, color='black', linewidth=1.5, *, fillColor=None, legend = False):
        self.outerPolygon.Plot(color, linewidth, linestyle=(5, (10, 1)), legend= 'Outer Polygon' if legend else None)
        
        show_legend = True
        for hole in self.innerPolygons:
            hole.Plot(color, linewidth, linestyle=(0, (3, 1, 1, 1)), legend='Inner Polygons (Hole)' if show_legend and legend else None)
            show_legend = False
        
        if legend:
            plt.legend()
        
        if fillColor is not None:
            self.FillArea(color=fillColor)
            
    def FillArea(self, color='lightgray'):
        x = self.outerPolygon.Vertices()[:,0]
        y = self.outerPolygon.Vertices()[:,1]
        plt.fill(x, y, color=color)
        for hole in self.innerPolygons:
            x = hole.Vertices()[:,0]
            y = hole.Vertices()[:,1]
            plt.fill(x, y, color='white')
        
        

        
        
        