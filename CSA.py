
import numpy as np
import random as rnd
from Classes.Shape import *

HEX_NUM = 100
UPDATE_MOVES_PERC = 25

# Lista de colores distinguibles en formato HEX
DISTINGUISHABLE_COLORS = [
    "#FF0000", "#0000FF", "#00CC00", "#FFFF00", "#00FFFF", "#FF00FF", "#FFA500",
    "#800080", "#FFC0CB", "#40E0D0", "#BFFF00", "#FFD700", "#8B4513", "#808080", "#000000"
]

####Celular Segregation Algorithm###
class CSA:
    def __init__(self, shape: Shape, group_percentage: List[float], *, radiusHexagons: float = None, totalHexagons: int = HEX_NUM): 
        self.shape: Shape = shape  
        self.group_percentage: List[float] = group_percentage

        self.radiusHexagons: float = 1/3*np.sqrt(2*np.sqrt(3)*shape.Area()/totalHexagons)
        if radiusHexagons is not None:
            self.radiusHexagons = radiusHexagons
        
        self.matrix = self._HexagoniseShape()  #matrix (hexagonal grid) with 0 outside the shape; 1, 2, 3, ... inside the shape and assigned to a group or cluster; and -1 inside the shape but not assigned 
        self._GroupAllocations()
        
    def _HexagoniseShape(self) -> np.ndarray[Any, np.dtype[np.int32]]: 
        # Get 0 0 position, loweer left corner
        shape_vertices = self.shape.outerPolygon.Vertices()
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf
        for vertex in shape_vertices:
            min_x = min(min_x, vertex[0])
            min_y = min(min_y, vertex[1])
            max_x = max(max_x, vertex[0])
            max_y = max(max_y, vertex[1])
        
        delta_x, delta_y = 0, 0
        if HEXAGON_ORIENTATION == 'FLAT':
            delta_x = 3/2*self.radiusHexagons
            delta_y = np.sqrt(3)/2*self.radiusHexagons
        else:
            delta_x = np.sqrt(3)/2*self.radiusHexagons
            delta_y = 3/2*self.radiusHexagons
        
        ROWS, COLUMNS = int((max_y-min_y)//delta_y + 1) , int((max_x-min_x)//delta_x + 1)
        matrix = np.full((ROWS, COLUMNS), 0)
        
        for j in range(0, ROWS, 2):
            for i in range(0, COLUMNS, 2):
                x, y  = min_x + i*delta_x,  min_y + j*delta_y
                center = np.array([x, y])
                if self.shape.PointIsInside(center):
                    matrix[j, i] = -1

                if i+1 < COLUMNS and j+1 < ROWS:
                    x, y = x+delta_x, y+delta_y
                    center = np.array([x, y])
                    if self.shape.PointIsInside(center):
                        matrix[j+1, i+1] = -1          
        
        return matrix
    
    def _GroupAllocations(self):
        # Verificar que los porcentajes sumen a 1 o a 100
        if not np.isclose(sum(self.group_percentage), 100):
            raise ValueError("Percentage areas must sum to 100")

        group_percentage = [p / 100 for p in self.group_percentage]

        total_hexagons = np.sum(self.matrix == -1)
        hexagons_per_color = [int(round(total_hexagons * p)) for p in group_percentage]
        
        while sum(hexagons_per_color) < total_hexagons:
            hexagons_per_color[np.argmax(group_percentage)] += 1
        while sum(hexagons_per_color) > total_hexagons:
            hexagons_per_color[np.argmax(hexagons_per_color)] -= 1

        allocations = []
        for numbers, count in enumerate(hexagons_per_color):
            allocations.extend([numbers+1] * count)
        rnd.shuffle(allocations)
        
        ROWS, COLUMNS = self.matrix.shape
        for j in range(0, ROWS):
            for i in range(0, COLUMNS):
                if self.matrix[j, i] != 0:
                    self.matrix[j, i] = allocations.pop()
                    
    
    def Plot(self):
        legend_elements = [f"group {i+1}: {percentage:.1f}%" for i, percentage in enumerate(self.group_percentage)]
        legend_ploted = [False for _ in range(len(self.group_percentage))]
        
        shape_vertices = self.shape.outerPolygon.Vertices()
        min_x, min_y = np.inf, np.inf
        max_x, max_y = -np.inf, -np.inf
        for vertex in shape_vertices:
            min_x = min(min_x, vertex[0])
            min_y = min(min_y, vertex[1])
            max_x = max(max_x, vertex[0])
            max_y = max(max_y, vertex[1])
            
        delta_x, delta_y = 0, 0
        if HEXAGON_ORIENTATION == 'FLAT':
            delta_x = 3/2*self.radiusHexagons
            delta_y = np.sqrt(3)/2*self.radiusHexagons
        else:
            delta_x = np.sqrt(3)/2*self.radiusHexagons
            delta_y = 3/2*self.radiusHexagons

        ROWS, COLUMNS = self.matrix.shape
        for j in range(0, ROWS):
            for i in range(0, COLUMNS):
                if self.matrix[j, i] != 0:
                    group = self.matrix[j, i]
                    x, y = min_x + i*delta_x, min_y + j*delta_y
                    center = np.array([x, y])
                    hexagon = Hexagon(center, self.radiusHexagons)
                    hexagon.polygon.Plot(color='black', linewidth=self.radiusHexagons, fillColor=DISTINGUISHABLE_COLORS[group-1], legend=legend_elements[group-1] if not legend_ploted[group-1] else None)
                    legend_ploted[group-1] = True
        
        plt.legend()

    def Resolve(self):
        check = True
        spatialSwap = self._SpatialSwap(self.matrix)
        swapCost = spatialSwap.Swap()
        while swapCost > 0:
            check = True
            swapCost = spatialSwap.Swap()
            if swapCost <= 0:
                # joinGroupMatrix = np.full(self.matrix.shape, 0)
                # for group in spatialSwap.groups:
                #     joinGroupMatrix[group.groupMatrix > 0] = group.groupId
                # self.matrix = joinGroupMatrix
                # self.Plot()
                # plt.gca().set_aspect('equal')
                # plt.show()
                check = True
                while swapCost <= 0 and check:
                    check = False
                    for group in spatialSwap.groups:
                        if len(group.clusterSize) > 1:
                            group.joinClusterCost *= 1.1
                            check = True
                        else:
                            group.joinClusterCost = 0.3
                    
                    spatialSwap.SetUpSwapMatrix()
                    swapCost = spatialSwap.Swap()
                        
                
                
            print(swapCost)
        joinGroupMatrix = np.full(self.matrix.shape, 0)
        for group in spatialSwap.groups:
            joinGroupMatrix[group.groupMatrix > 0] = group.groupId
        self.matrix = joinGroupMatrix
        self.Plot()
        plt.gca().set_aspect('equal')
        plt.show()
        
    

    
        
    class _SpatialSwap:
        def __init__(self, matrix: np.ndarray[Any, np.dtype[np.int32]]):
            
            self.swapMatrix = np.full((int(matrix.shape[0]*2-1), int(matrix.shape[1]*2-1), 2), -np.inf) #3rd dimension for 3 weights for swapping -> 0: Distance bewteen cluster centroids; 1: Compactness; 2: Frontier
            
            #SETTUP GROUPS CLUSTERS
            self.groups: List[self._Group] = [] #List of Groups
            for group in range(1, np.max(matrix)+1):
                group = self._Group(group, matrix)
                self.groups.append(group)
            
            self.tabuStartTime = 30
            self.tabuPositions, self.tabuTimes = [], []      
            #SETTUP SWAPS
            self.SetUpSwapMatrix()
               
               
            
                        
        def SetUpSwapMatrix(self):
            print("-------------------------------------------------------------------------------------")
            self.swapMatrix = np.full(self.swapMatrix.shape, -np.inf) 
            for n in range(len(self.tabuPositions)-1, -1, -1):
                self.tabuTimes[n] -= 3
                j, i = self.tabuPositions[n]
                self.swapMatrix[j, i] = np.array([-np.inf, -np.inf]) 
                if self.tabuTimes[n] <= 0:
                    self.tabuTimes.pop(n)
                    self.tabuPositions.pop(n)
                    self.swapMatrix[j, i] = np.array([np.inf, np.inf]) 
            for group in self.groups:
                self.swapMatrix[group.GroupClusters() == np.inf] = np.array([np.inf, np.inf])
            self.RecalculateSwapCost()
            
            
            
        def _SwapCost(self, swapPosition: Tuple[int, int]) -> np.ndarray[Any, np.dtype[np.float64]]:  #swapPosition: (j, i), impoisible swap: -np.inf
            #FIND AFFECTED POSITIONS AND CLUSTERS
            affectedPositions = SwapOrientation(swapPosition)
            if not(0<= affectedPositions[0][0] < (self.swapMatrix.shape[0]+1)//2 and 0<= affectedPositions[0][1] < (self.swapMatrix.shape[1]+1)//2 and 0<= affectedPositions[1][0] < (self.swapMatrix.shape[0]+1)//2 and 0<= affectedPositions[1][1] < (self.swapMatrix.shape[1]+1)//2):
                return np.array([-np.inf, -np.inf]) 
            
            affectedGroups = np.array([0, 0])
            for n, position in enumerate(affectedPositions):
                j, i = position
                for group in self.groups:
                    if group.groupMatrix[j, i] > 0:
                        affectedGroups[n] = group.groupId
                        break
            
            if affectedGroups[0] == affectedGroups[1] or affectedGroups[0]*affectedGroups[1]==0: #same group or outside the shape
                return np.array([-np.inf, -np.inf]) 
            
            #CALCULATE SWAP
            totalCost = np.array([0., 0.])
            for n, group in enumerate(affectedGroups):
                group = self.groups[group-1]
                originPosition, changePosition = affectedPositions[n], affectedPositions[1-n]
                
                cost = group.MoveCost(originPosition, changePosition)
                totalCost += cost
                
            return totalCost    
            
        def Swap(self):
            weighted_sum = (5*self.swapMatrix[:, :, 0]+5*self.swapMatrix[:, :, 1]) 
            swapPosition, max_value = np.unravel_index(np.argmax(weighted_sum), self.swapMatrix.shape[:2]), np.max(weighted_sum)
            if max_value <= 0:
                return 0
            #FIND AFFECTED POSITIONS AND CLUSTERS
            affectedPositions = SwapOrientation(swapPosition)
            # print(affectedPositions)
            affectedGroups = np.array([0, 0])
            for n, position in enumerate(affectedPositions):
                j, i = position
                for group in self.groups:
                    if group.groupMatrix[j, i] > 0:
                        affectedGroups[n] = group.groupId
                        break
            #MOVE BOTH POSITIONS
            for n, groupId in enumerate(affectedGroups):
                group = self.groups[groupId-1]
                originPosition, changePosition = affectedPositions[n], affectedPositions[1-n]
                deltaCost = group.Move(originPosition, changePosition)
                self.swapMatrix[deltaCost == np.inf] = np.array([np.inf, np.inf])
            
            #RECALCULATE AROUND CELL SWAPS
            for afPos in affectedPositions:
                for aroundPositions in Neighbors():
                    j, i = afPos[0]+aroundPositions[0], afPos[1]+aroundPositions[1]
                    for dj, di in Neighbors():
                        nj, ni = j*2 + dj, i*2 + di
                        if 0 <= nj < self.swapMatrix.shape[0] and 0 <= ni < self.swapMatrix.shape[1]:
                            self.swapMatrix[nj, ni] = np.array([np.inf, np.inf]) 
            self.swapMatrix[swapPosition[0], swapPosition[1]] = np.array([-np.inf, -np.inf])
            
            for n in range(len(self.tabuPositions)-1, -1, -1):
                self.tabuTimes[n] -= 1
                j, i = self.tabuPositions[n]
                self.swapMatrix[j, i] = np.array([-np.inf, -np.inf]) 
                if self.tabuTimes[n] == 0:
                    self.tabuTimes.pop(n)
                    self.tabuPositions.pop(n)
                    self.swapMatrix[j, i] = np.array([np.inf, np.inf]) 
            self.tabuPositions.append(swapPosition)
            self.tabuTimes.append(self.tabuStartTime)
                
            self.RecalculateSwapCost()
            return max_value
        
        def RecalculateSwapCost(self):
            for n in range(len(self.tabuPositions)-1, -1, -1):
                j, i = self.tabuPositions[n]
                self.swapMatrix[j, i] = np.array([-np.inf, -np.inf]) 
            for j, i in np.argwhere(self.swapMatrix[:,:,0] == np.inf):
                self.swapMatrix[j, i] = self._SwapCost((j, i))  
            
                
        class _Group:
            def __init__(self, groupId: int, matrix: np.ndarray[Any, np.dtype[np.int32]]):
                self.groupId = groupId
                self.groupMatrix = matrix.copy()
                self.groupMatrix[(self.groupMatrix > 0) & (self.groupMatrix != self.groupId)] = -1
                
                self.swapMatrix = np.full((int(matrix.shape[0]*2-1), int(matrix.shape[1]*2-1)), 0) #0: No swap; 1: Swap 
                for j, i in np.argwhere(self.groupMatrix == self.groupId):
                    for dj, di in Neighbors():
                        nj, ni = j + dj, i + di
                        if 0 <= nj < self.groupMatrix.shape[0] and 0 <= ni < self.groupMatrix.shape[1]:
                            if self.groupMatrix[nj, ni] == -1:
                                self.swapMatrix[j*2+dj, i*2+di] = 1
                
                self.centroid = np.mean(np.argwhere(self.groupMatrix == self.groupId), axis=0)          
                
                self.maxDistanceCost = np.sqrt(matrix.shape[0]**2 + matrix.shape[1]**2)
                self.totalCells, self.clusterSize  = len(np.argwhere(self.groupMatrix > 0)), []
                self.numMoves = 0
                self.joinClusterCost = 0.3
                
                
            def GroupClusters(self) -> Tuple[np.ndarray[Any, np.dtype[np.int32]], int]:
                neighbors_deltas = Neighbors()
                
                ROWS, COLUMNS = self.groupMatrix.shape
                mask = self.groupMatrix > 0  # Máscara booleana para marcar celdas sin procesar
                cluster_id = 1
                cluster_sizes = {}  # Guardar tamaños de cada cluster
                self.clusterSize = 0
                # Recorrer todas las posiciones con grupo (> 0)
                for j, i in np.argwhere(mask):
                    if mask[j, i]:
                        stack = [(j, i)]
                        cluster_count = 0
                        mask[j, i] = False  # Marcar como visitado
                        while stack:
                            cj, ci = stack.pop()
                            self.groupMatrix[cj, ci] = cluster_id
                            cluster_count += 1
                            # Revisar vecinos
                            for dj, di in neighbors_deltas:
                                nj, ni = cj + dj, ci + di
                                if 0 <= nj < ROWS and 0 <= ni < COLUMNS and mask[nj, ni]:
                                    stack.append((nj, ni))
                                    mask[nj, ni] = False
                        cluster_sizes[cluster_id] = cluster_count
                        if 100 * cluster_count / self.totalCells < 50:
                            self.clusterSize +=1
                        cluster_id += 1

                # Ordenar clusters de mayor a menor tamaño y crear el mapeo de IDs
                sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
                cluster_map = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_clusters, start=1)}

                # Actualizar la matriz de grupos utilizando indexación booleana
                for old_id, new_id in cluster_map.items():
                    self.groupMatrix[self.groupMatrix == old_id] = new_id
                
                self.clusterSize =  [False]*(len(sorted_clusters)-self.clusterSize)+[True]*self.clusterSize
                
                deltaCost = np.full(self.swapMatrix.shape, 0, float)
                deltaCost[self.swapMatrix == 1] = np.inf
                return deltaCost
                        


                
            
            def MoveCost(self, originPosition, movePosition):
                
                distanceCost = (np.linalg.norm(self.centroid - originPosition)-np.linalg.norm(self.centroid - movePosition))*np.linalg.norm(self.centroid - originPosition)/self.maxDistanceCost
                if self.clusterSize[self.groupMatrix[originPosition[0], originPosition[1]]-1]:
                    distanceCost += self.joinClusterCost
                else:
                    distanceCost *= 1.3
                    
                froniterCost = -2 
                j, i = originPosition
                for dj, di in Neighbors():
                    nj, ni = j + dj, i + di
                    if 0 <= nj < self.groupMatrix.shape[0] and 0 <= ni < self.groupMatrix.shape[1]:
                        if self.groupMatrix[nj, ni] > 0:
                            froniterCost -= 1
                        elif self.groupMatrix[nj, ni] == -1:
                            froniterCost += 1
                j, i = movePosition
                for dj, di in Neighbors():
                    nj, ni = j + dj, i + di
                    if 0 <= nj < self.groupMatrix.shape[0] and 0 <= ni < self.groupMatrix.shape[1]:
                        if self.groupMatrix[nj, ni] > 0:
                            froniterCost += 1
                        elif self.groupMatrix[nj, ni] == -1:
                            froniterCost -= 1
                            
                return np.array([round(distanceCost, 2), round(froniterCost/6, 2)])  
            
            def Move(self, originPosition, movePosition):
                
                j, i = originPosition
                clusterId = self.groupMatrix[j, i]
                for dj, di in Neighbors():
                    nj, ni = j + dj, i + di
                    if 0 <= nj < self.groupMatrix.shape[0] and 0 <= ni < self.groupMatrix.shape[1]:
                        if self.groupMatrix[nj, ni] > 0:
                            self.swapMatrix[j*2+dj, i*2+di] = 1
                        elif self.groupMatrix[nj, ni] == -1:
                            self.swapMatrix[j*2+dj, i*2+di] = 0
                self.groupMatrix[j, i] = -1
                
                j, i = movePosition
                for dj, di in Neighbors():
                    nj, ni = j + dj, i + di
                    if 0 <= nj < self.groupMatrix.shape[0] and 0 <= ni < self.groupMatrix.shape[1]:
                        if self.groupMatrix[nj, ni] == -1:
                            self.swapMatrix[j*2+dj, i*2+di] = 1
                        elif self.groupMatrix[nj, ni] > 0:
                            self.swapMatrix[j*2+dj, i*2+di] = 0
                            if self.groupMatrix[nj, ni] < clusterId:
                                clusterId = self.groupMatrix[nj, ni]
                self.groupMatrix[j, i] = clusterId
                
                self.numMoves += 1
                deltaCost = np.full(self.swapMatrix.shape, 0, float)
                if self.numMoves == int(self.totalCells*UPDATE_MOVES_PERC/100):
                    self.centroid = np.mean(np.argwhere(self.groupMatrix > 0), axis=0)      
                    deltaCost = self.GroupClusters()
                    self.numMoves = 0
                        
                return deltaCost
                    
                            

  
        

def Matrix_to_String(matrix: np.ndarray[Any, np.dtype[np.int32]]) -> str:
    string = ""
    for row in matrix[::-1]:
        formatted_row = " ".join([
            f"{cell:6.2f}" if isinstance(cell, float) else f"{cell:4d}" 
            for cell in row
        ])
        string = string + "\n" + formatted_row
    return string
    

    


