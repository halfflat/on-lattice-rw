import enum
import math
import numpy as np

class PeriodicNet:
    def __init__(self, edges, positions, translations):
        # edges is an integer numpy array with one row per edge.
        # Each row has the format:
        #     [ vertex-index x1 x2 ... xn neighbour-index]
        # where the vertex indices are 0-based and x1, x2, ...
        # are the translations to the corresponding neighbouring
        # unit cell. Edges are in order of vertex-index.

        self.edges = edges.copy()
        self.positions = positions.copy()
        self.translations = translations.copy()

        self.n_vertices = edges[:,0].max()+1
        self.degree =  np.uint32(edges.shape[0]/self.n_vertices)
        self.dimension = edges.shape[1]-2

        if (positions.shape != (self.n_vertices, self.dimension)):
            raise Exception("Bad position array")

        if (translations.shape != (self.dimension, self.dimension)):
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
        np.array([[0, (1+_r3)/_r2], [(1+_r3)/_r2, 0]]))

# Hexagonal regular 6-net
hxl = PeriodicNet(
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, -1, 1, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 1, -1, 0]]),
        np.array([[0, 0]]),
        np.array([[1, 0], [0.5, _r3/2]]))


# Predefined 3-periodic nets
# TODO:
# Laves graph regular 3-net srs
# Diamond regular 4-net dia
# NbO regular 4-net nbo
# Cubic regular 6-net pcu
# Body-centred cubic regular 8-net bcu
# Face-centred cubic quasiregular 12-net fcu

class Domain:
    # Given a PeriodicNet _net_, a tuple _shape_ describing the extent, and a
    # _maximum occupancy_ per vertex, create the domain over which the walk is
    # perfomed.
    #
    # The extent tuple must have the same dimension as the periodic net.
    #
    # The domain is represented by an array with extents (I+2, J+2, K+2, V, M)
    # in the 3-dimensional case, given a shape tuple (I, J, K), a net with V
    # vertices, and a maximum occupancy M. The spatial boundary (in the I, J, K
    # dimensions) is marked with special values indicating that the vertices
    # are in fact boundary values.
    #
    # Domain access is by coordinates (i, j, k, v) (for 3-dimensional case) or
    # (i, j, v) (2-dimensional case) where (i, j, k) are the cell-coordinates
    # and v is the vertex-index. These coordinates are offset by the boundary
    # padding, that is, i, j, and k are incremented before accessing the
    # internal domain array.
    #
    # The array holds 32-bit values representing the (packed) information
    # describing each occupant.

    BORDER = np.invert(np.uint32(0))
    OCCUPANT_DTYPE = np.dtype([('nid', np.uint32), ('occupancy', np.uint8), ('prev', np.uint8)])

    class Occupancy(enum.IntEnum):
        EMPTY = 0
        ODD = 1
        EVEN = 2
        FULL = 3

    def __init__(self, net, shape, max_occupancy):
        self._mo = max_occupancy;
        self._net = net
        if self._net.dimension != len(shape):
            raise Exception("shape is wrong dimension")

        if self._net.degree>64:
            raise Exception("only supporting up to degree 64 with packed representation")

        # make domain but fill boundary with boundary marker
        inner = shape + (self._net.n_vertices, self._mo)
        pad = [(1,1)]*len(inner)
        pad[-2:] = [(0,0), (0,0)]

        self._pad_offset = np.ones((len(inner),), dtype=np.int32)
        self._pad_offset[-2:] = [0, 0]

        self._domain = np.pad(np.zeros(inner, dtype=np.uint32), pad, mode='constant', constant_values=Domain.BORDER)

    # Return extent, excluding the padded boundaries and occupancy axis.
    def extent():
        return (_domain.shape() - _pad_offset*2)[:-1]

    def max_occupancy():
        return self._mo

    def unpack(packed):
        u = np.recarray(packed.shape, dtype=Domain.OCCUPANT_DTYPE)
        u.nid = np.right_shift(packed, 8)
        u.occupancy = np.right_shift(np.bitwise_and(packed, 255), 6)
        u.prev = np.bitwise_and(packed, 63)
        return u

    def pack(unpacked):
        return np.array(np.left_shift(unpacked.nid, 8) + np.left_shift(unpacked.occupancy, 6) + unpacked.prev, dtype=np.uint32)

    # Return occupant in given slot in the domain at vertex, or all occupants at vertex if slot is None.
    def get(self, vertex, slot = None):
        if isinstance(vertex, tuple): vertex = np.array(vertex, dtype=np.uint32)
        index = vertex + self._pad_offset[:-1]
        if slot is None:
            return self._domain[tuple(index)+(slice(None),)]
        else:
            return self._domain[tuple(index)+(slot,)]

    # Update occupant at slot in the domain at vertex with packed value, or all occupants with array of packed values if slot is None.
    def put(self, vertex, occupant, slot = None):
        if isinstance(vertex, tuple): vertex = np.array(vertex, dtype=np.uint32)
        index = vertex + self._pad_offset[:-1]
        if slot is None:
            self._domain[tuple(index)+(slice(None),)] = occupant
        else:
            self._domain[tuple(index)+(slot,)] = occupant

    def traverse(self, vertex, edge_index):
        if isinstance(vertex, tuple): vertex = np.array(vertex, dtype=np.uint32)
        u = vertex + self._net.edges[edge_index,1:-1]
        u[-1] = self._net.edges[edge_index,-1]
        return u

    def position(self, vertex):
        if isinstance(vertex, tuple): vertex = np.array(vertex, dtype=np.uint32)
        return self._net.positions[vertex[-1]] + np.matmul(vertex[:self._net.dimension], self._net.translations)

    def is_border(self, vertex):
        return get(self, vertex, 0)==BORDER

