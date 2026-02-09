import math
import numpy as np

class Polygon:
    def __init__(self, points = [], validateConvex = True):
        self.verticies = np.array(points)
        self.numVerticies = len(self.verticies)

        if validateConvex:
            if not self.is_convex():
                raise ValueError("Verticies do not form a convex polygon.")

    def is_convex(self):
        if self.numVerticies < 3:
            return False

        sign = None

        for i in range(self.numVerticies):
            #Three consecutive points
            p1 = self.verticies[i]
            p2 = self.verticies[(i + 1) % self.numVerticies]
            p3 = self.verticies[(i + 2) % self.numVerticies]

            #Vectors
            v1 = p2 - p1
            v2 = p3 - p2

            #Cross product
            cross = v1[0] * v2[-1] - v1[1] * v2[0]

            if abs(cross) < 1e-10:      #Nearly collinear
                continue

            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False            #If sign changed, it's not convex

        return True

    def printPolygon(self):
        print(self.verticies)

    def calcDistance(self, point1Indx, point2Indx):
        point1 = self.verticies[point1Indx]
        point2 = self.verticies[point2Indx]
        return np.linalg.norm(point2 - point1)

    def getAllDistances(self):
        #Gets distances between all points
        distances = {}
        for i in range (self.numVerticies):
            for j in range(i + 1, self.numVerticies):
                distances[(i, j)] = self.calcDistance(i, j)

        return distances

    def getAllEdges(self):
        #Gets distances of all edges
        edges = {}
        for i in range (self.numVerticies):
            j = (i + 1) % self.numVerticies
            edges[(i,j)] = self.calcDistance(i, j)

        return edges

    def getAllDiagonals(self):
            #Gets distances of all diagonals
            diagonals = {}
            for i in range(self.numVerticies):
                for j in range(i + 2, self.numVerticies):
                    if not (i == 0 and j == self.numVerticies - 1):
                        diagonals[(i,j)] = self.calcDistance(i, j)

            return diagonals


    def _diagonals_cross(self, diagonal1, diagonal2):
        # Check if 2 diagonals with verticies (i,j) and (k, l) cross each other
        i, j = diagonal1
        k, l = diagonal2

        if i > j:
            i, j = j, i
        if k > l:
            k, l = l, k

        #Check if they share a vertex
        if i == k or i == l or j == k or j == l:
            return False

        #Check crossing condition
        return (i < k < j < l) or (k < i < l < j)

    def _crosses_any(self, i, j, existing_diagonals):
        #Check if a diagonal (i,j) crosses any existing diagonals
        for diagonal in existing_diagonals:
            if self._diagonals_cross((i, j), diagonal):
                return True

        return False

    def greedy_triangulation(self):
        #Finds triangulation using greedy algorythm

        #Get all possible diagonals
        allDiagonalsDict = self.getAllDiagonals()

        #Convert to list of (i, j, distance) and sort it by distance
        allDiagonals = [(i, j, distance) for (i, j), distance in allDiagonalsDict.items()]
        allDiagonals.sort(key=lambda x : x[2])

        #Greedy non-crossing diagonals
        triangulation  = []
        totalLength = 0

        for i, j, dist in allDiagonals:
            if not self._crosses_any(i,j, triangulation):
                triangulation.append((i, j))
                totalLength += dist

                if len(triangulation) == self.numVerticies - 3:
                    break

        return triangulation, totalLength

    def visualize_triangulation(self, triangulation):
        """Print triangulation results."""
        print(f"Triangulation of {self.numVerticies}-gon:")
        print(f"Number of diagonals: {len(triangulation)}")
        print(f"Expected: {self.numVerticies - 3}")
        print("\nDiagonals:")
        for i, (v1, v2) in enumerate(triangulation, 1):
            dist = self.calcDistance(v1, v2)
            print(f"  {i}. Vertex {v1} → Vertex {v2}: {dist:.4f}")

    def all_triangulation(self):
        #Find all possible triangulations of a polygon, recursively
        if self.numVerticies <= 3:
            return [[]]

        memo = {}

        def triangulate_subpolygon(verticies):
            key = tuple(verticies)

            #Check memo
            if key in memo:
                return memo[key]

            if len(verticies) <= 3:
                return [[]]

            first, last = verticies[0], verticies[-1]
            allTriangulations = []

            #Try each middle vertex k to form a triangle with (first, k, last)
            for i in range(1, len(verticies) - 1):
                k = verticies[i]

                #Determine which diagonals are needed for this triangle
                currentDiagonals = []

                #Add diagonal (first, k) if k is not adjacent to first
                if i > 1:
                    currentDiagonals.append(tuple(sorted([first, k])))

                #Add diagonal (k, last) if k is not adjacent to last
                if i < len(verticies) - 2:
                    currentDiagonals.append(tuple(sorted([k, last])))

                #Recursively triangulate the left and right sub-polygons
                leftVerticies = verticies[ : i + 1]
                rightVerticies = verticies[i : ]


                leftTriangulations = triangulate_subpolygon(leftVerticies)
                rightTriangulations = triangulate_subpolygon(rightVerticies)

                #Combine all possibilities
                for left in leftTriangulations:
                    for right in rightTriangulations:
                        combined = currentDiagonals + left + right
                        allTriangulations.append(combined)

            memo[key] = allTriangulations
            return allTriangulations

        #Start with the full polygon
        vertexList = list(range(self.numVerticies))
        return triangulate_subpolygon(vertexList)

    def compare_triangulations(self):
        #Compare greedy triangulations with all triangulations to find an optimal solution
        greedyDiagonals, greedyWeight = self.greedy_triangulation()

        allTriangulations = self.all_triangulation()

        print(f"{'=' * 60}")
        print(f"TRIANGULATION COMPARISON FOR {self.numVerticies}-GON")
        print(f"{'=' * 60}")
        print(f"\nTotal possible triangulations: {len(allTriangulations)}")
        print(f"Expected (Catalan number C({self.numVerticies - 2})): {len(allTriangulations)}")


        #Calculate the weight for each triangulation
        triangulationWeights = []
        for triangulation in allTriangulations:
            totalWeight = sum(self.calcDistance(i, j) for i, j in triangulation)
            triangulationWeights.append((triangulation, totalWeight))

        #Sort by weight
        triangulationWeights.sort(key=lambda x: x[1])

        #Find optimal
        optimalDiagonals, optimalWeight = triangulationWeights[0]

        # Display all triangulations
        print(f"\n{'=' * 60}")
        print("ALL TRIANGULATIONS (sorted by total weight):")
        print(f"{'=' * 60}")
        for i, (diags, weight) in enumerate(triangulationWeights, 1):
            marker = ""
            if set(diags) == set(greedyDiagonals):
                marker = " ← GREEDY"
            if weight == optimalWeight:
                marker += " ← OPTIMAL" if marker else " ← OPTIMAL"

            print(f"\n{i}. Weight: {weight:.4f}{marker}")
            print(f"   Diagonals: {diags}")

        # Summary
        print(f"\n{'=' * 60}")
        print("SUMMARY:")
        print(f"{'=' * 60}")
        print(f"Greedy algorithm weight: {greedyWeight:.4f}")
        print(f"Optimal weight:          {optimalWeight:.4f}")

        if greedyWeight == optimalWeight:
            print("✓ Greedy found the optimal solution!")
        else:
            difference = greedyWeight - optimalWeight
            percent = (difference / optimalWeight) * 100
            print(f"✗ Greedy is {difference:.4f} longer ({percent:.2f}% worse)")

        return {
            'all_triangulations': triangulationWeights,
            'greedy': (greedyDiagonals, greedyWeight),
            'optimal': (optimalDiagonals, optimalWeight)
        }