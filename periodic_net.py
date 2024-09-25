import enum
import math
import numbers
import numpy as np

class PeriodicNet:
    # represents a periodic net of uniform degree (i.e. regular as a graph).

    def __init__(self, edges, positions, translations):
        # edges is an integer numpy array with one row per edge.
        # Each row has the format:
        #     [ vertex-index x1 x2 ... xn neighbour-index]
        # where the vertex indices are 0-based and x1, x2, ...
        # are the translations to the corresponding neighbouring
        # unit cell. Edges are in order of vertex-index.

        self.edges = np.asarray(edges, dtype=np.int32)
        self.positions = np.asarray(positions, dtype=np.double)
        self.translations =np.asarray(translations, dtype=np.double)

        self.n_vertices = edges[:,0].max()+1
        self.degree =  np.uint32(edges.shape[0]/self.n_vertices)
        self.dimension = edges.shape[1]-2

        if (self.positions.shape != (self.n_vertices, self.dimension)):
            raise Exception("Bad position array")

        if (self.translations.shape != (self.dimension, self.dimension)):
            raise Exception("Bad position array")

        # compute inverse edge map from vertex 0; then verify
        # for other points.

        n_edges = edges.shape[0]
        self.inv_edges = np.empty((n_edges,),dtype=np.int32)
        self.inv_edges.fill(-1)

        for r in range(0, n_edges):
            ni = edges[r,1]
            ie = edges[r,:].copy()
            ie[[0,-1]] = ie[[-1,0]]
            ie[1:(1+self.dimension)] *= -1
            # neighbour vertex is ie[0]; search corresponding span of edges
            for s in range(ie[0]*self.degree, (ie[0]+1)*self.degree):
                if np.array_equal(ie, edges[s,:]):
                    self.inv_edges[r] = s
                    break
            if self.inv_edges[r]==-1:
                raise Exception("couldn't find inverse edge")

    def edges_from(self, vertex_index):
        offset = vertex_index*self.degree;
        return self.edges[offset:offset+self.degree]

    def edge_indices_from(self, vertex_index):
        offset = vertex_index*self.degree;
        return np.arange(offset, offset+self.degree, dtype=np.uint32)

    # Below, coord is a tuple or array denoting a vertex in the tesselation.
    # coord is of length dimension+1; the last component denotes the vertex index.

    def position(self, coord):
        coord = np.asarray(coord)
        return self.positions[coord[-1]] + np.matmul(coord[:self.dimension], self.translations)

    def traverse(self, coord, edge_index):
        u = np.asarray(coord) + self.edges[edge_index,1:]
        u[-1] = self.edges[edge_index,-1]
        return u

    # return u s.t. traverse(u, edge_index) == coord
    def traverse_r(self, coord, edge_index):
        return self.traverse(coord, self.inv_edges[edge_index])

# Predefined 2-periodic nets

_r3 = math.sqrt(3)
_r2 = math.sqrt(2)

# Honeycomb regular 3-net
hcb = PeriodicNet(
        np.array([[0, 0, 0, 1], [0, 0, -1, 1], [0, -1, 0, 1],
                  [1, 0, 0, 0], [1, 0,  1, 0], [1,  1, 0, 0]]),
        np.array([[0.5, 0.5/_r3], [1, 1.0/_r3]]),
        np.array([[1, 0], [0.5, _r3/2]]))

# Square regular 4-net
sql = PeriodicNet(
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, -1, 0]]),
        np.array([[0, 0]]),
        np.array([[1, 0], [0, 1]]))

# Triangle-triangle-square irregular 5-net
tts = PeriodicNet(
        np.array([[0, 0, 0, 1], [0, 0, 0, 3], [0, -1,  0, 1], [0,  0, -1, 3], [0,  0, -1, 2],
                  [1, 0, 0, 2], [1, 0, 0, 0], [1,  0, -1, 2], [1,  1,  0, 0], [1,  1,  0, 3],
                  [2, 0, 0, 3], [2, 0, 0, 1], [2,  1,  0, 3], [2,  0,  1, 1], [2,  0,  1, 0],
                  [3, 0, 0, 0], [3, 0, 0, 2], [3,  0,  1, 0], [3, -1,  0, 2], [3, -1,  0, 1]]),
        np.array([[      _r3/(2*_r2),         1/(2*_r2)],
                  [(1+2*_r3)/(2*_r2),       _r3/(2*_r2)],
                  [  (2+_r3)/(2*_r2), (1+2*_r3)/(2*_r2)],
                  [        1/(2*_r2),   (2+_r3)/(2*_r2)]]),
        np.array([[(1+_r3)/_r2, 0], [0, (1+_r3)/_r2]]))

# Hexagonal regular 6-net
hxl = PeriodicNet(
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, -1, 1, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 1, -1, 0]]),
        np.array([[0, 0]]),
        np.array([[1, 0], [0.5, _r3/2]]))

