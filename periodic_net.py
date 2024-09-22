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
        return traverse(self, coord, self.inv_edges[edge_index])

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


# Predefined 3-periodic nets
# TODO:
# Laves graph regular 3-net srs
# Diamond regular 4-net dia
# NbO regular 4-net nbo
# Cubic regular 6-net pcu
# Body-centred cubic regular 8-net bcu
# Face-centred cubic quasiregular 12-net fcu

#class Domain:
#    # Given a PeriodicNet _net_, a tuple _shape_ describing the extent, and a
#    # _maximum occupancy_ per vertex, create the domain over which the walk is
#    # perfomed.
#    #
#    # The extent tuple must have the same dimension as the periodic net.
#    #
#    # The domain is represented by an array with extents (I+2, J+2, K+2, V, M)
#    # in the 3-dimensional case, given a shape tuple (I, J, K), a net with V
#    # vertices, and a maximum occupancy M. The spatial boundary (in the I, J, K
#    # dimensions) is marked with special values indicating that the vertices
#    # are in fact boundary values.
#    #
#    # Domain access is by coordinates (i, j, k, v) (for 3-dimensional case) or
#    # (i, j, v) (2-dimensional case) where (i, j, k) are the cell-coordinates
#    # and v is the vertex-index. These coordinates are offset by the boundary
#    # padding, that is, i, j, and k are incremented before accessing the
#    # internal domain array.
#    #
#    # The array holds 32-bit values representing the (packed) information
#    # describing each occupant: the neuron id (24 bits); occupancy class
#    # (2 bits); and previous edge index (6 bits).
#
#    BORDER = np.invert(np.uint32(0))
#    NOEDGE = np.uint8(63)
#    OCCUPANT_DTYPE = np.dtype([('nid', np.uint32), ('occupancy', np.uint8), ('prev', np.uint8)])
#
#    class Occupancy(enum.IntEnum):
#        EMPTY = 0
#        ODD = 1
#        EVEN = 2
#        FULL = 3
#
#        def flip(self):
#            match self:
#                case Occupancy.EVEN: return Occupancy.ODD
#                case Occupancy.ODD:  return Occupancy.EVEN
#                case _: return self
#
#
#    def __init__(self, net, shape, max_occupancy, priority):
#        """
#        Create and initialize a domain for lattice random walk
#
#        Keyword arguments
#        net              -- a PeriodicNet object describing the lattice
#        shape            -- tuple (I, J) or (I, J, K) describing extent of
#                            domain in lattice cells
#        max_occupancy    -- maximum number of residents per lattice site
#        priority         -- function p(nid, vtx_hash) that returns the
#                            numerical priority of neuron nid coming from the
#                            vertex represented by the integer vtx_hash
#        """
#
#        self._mo = max_occupancy;
#        self._net = net
#        slef._priority_fn = priority
#
#        if self._net.dimension != len(shape):
#            raise Exception("shape is wrong dimension")
#
#        if self._net.degree>63:
#            raise Exception("only supporting up to degree 63 with packed representation")
#
#        # make domain but fill boundary with boundary marker
#        inner = shape + (self._net.n_vertices, self._mo)
#        pad = [(1,1)]*len(inner)
#        pad[-2:] = [(0,0), (0,0)]
#
#        self._pad_offset = np.ones((len(inner),), dtype=np.int32)
#        self._pad_offset[-2:] = [0, 0]
#
#        self._domain = np.pad(np.zeros(inner, dtype=np.uint32), pad, mode='constant', constant_values=Domain.BORDER)
#
#        # pre-calculate offsets, shifts for computing vertex index hashes
#        self._vext = self._domain.shape[:-1]
#        self._vdim = len(self._vext)
#        self._hw = np.asarray([0]+[int(k-1).bit_length() for k in self._vext])
#        self._ho = self._hw.cumsum()
#
#    # Return extent, excluding the padded boundaries and occupancy axis.
#    extent = property(lambda self: (self._domain.shape - self._pad_offset*2)[:-1])
#
#    # Return number of occupancy slots per vertex
#    max_occupancy = property(lambda self: self._mo)
#
#    def priority(self, nid, vertex_or_hash):
#        if isinstance(vertex_or_hash, numbers.Integral):
#            return self._priority_fn(nid, vertex_or_hash)
#        else:
#            return self._priority_fn(nid, self.vertex_hash(vertex_or_hash))
#
#    def unpack(packed):
#        u = np.recarray(packed.shape, dtype=Domain.OCCUPANT_DTYPE)
#        u.nid = np.right_shift(packed, 8)
#        u.occupancy = np.right_shift(np.bitwise_and(packed, 255), 6)
#        u.prev = np.bitwise_and(packed, 63)
#        return u
#
#    def pack(unpacked):
#        return np.array(np.left_shift(unpacked.nid, 8) + np.left_shift(unpacked.occupancy, 6) + unpacked.prev, dtype=np.uint32)
#
#    # Convert vertex index to and from packed bit representation
#    def vertex_hash(self, vertex):
#        return (np.asarray(vertex) << self._ho[:-1]).sum()
#
#    def vertex_unhash(self, h):
#        return (h >> self._ho[:-1]) & ((1 << self._hw[1:]) - 1)
#
#    vertex_hash_bits = property(lambda self: self._ho[-1])
#
#    # Return occupant in given slot in the domain at vertex, or all occupants at vertex if slot is None.
#    def get(self, vertex, slot = None):
#        index = np.asarray(vertex) + self._pad_offset[:-1]
#        if slot is None:
#            return self._domain[tuple(index)+(slice(None),)]
#        else:
#            return self._domain[tuple(index)+(slot,)]
#
#    # Update occupant at slot in the domain at vertex with packed value, or all occupants with array of packed values if slot is None.
#    def put(self, vertex, occupant, slot = None):
#        index = np.asarray(vertex) + self._pad_offset[:-1]
#        if slot is None:
#            self._domain[tuple(index)+(slice(None),)] = occupant
#        else:
#            self._domain[tuple(index)+(slot,)] = occupant
#
#    def traverse(self, vertex, edge_index):
#        return self._net.traverse(vertex, edge_index)
#
#    def traverse_r(self, vertex, edge_index):
#        return self._net.traverse_r(vertex, edge_index)
#
#    def position(self, vertex):
#        return self._net.position(vertex)
#
#    def is_border(self, vertex):
#        index = np.asarray(vertex) + self._pad_offset[:-1]
#        return self.get(vertex, 0)==self.BORDER
#
#    def hosts(self, vertex, nid, parity):
#        """
#        Return True if site at vertex holds nid unprovisionally.
#
#        Test if any slot has matching nid with occupancy FULL or opposite of _parity_.
#        """
#
#        up = Domain.unpack(self.get(vertex))
#        return ((up.nid == nid) & ((up.occupancy == Domain.OCCUPANCY.FULL) | (up.occupancy == self.flip(parity)))).any()
#
#    def occupy(self, vertex, nid, parity, prev):
#        """
#        Provisionally place (nid, parity, prev) in a slot at vertex.
#
#        A placement will succeed if there is an EMPTY slot or a provisional slot, viz. a slot with the same occupancy _parity_, with
#        lower priority. A vertex can host at most one instance of a neuron nid so if nid is already present, placement can only
#        succeed if that instance is provisional and of lower priority.
#
#        Returns None on failure, or else a pair (evicted, slot index) where _evicted) is the id of a neuron that was removed to make
#        room for this neuron (or zero, if the slot was empty) and _slot index_ indicates with occupancy slot was occupied.
#        """
#
#        up = Domain.unpack(self._domain.get(vertex))
#
#        slot = None   # or (slot index, slot priority)
#
#        for j in range(self._mo):
#            if up[j].nid == nid:
#                if up[j].occupancy != parity:
#                    return None
#                else:
#                    slot = (j, self.priority(up[j].nid, self.traverse_r(vertex, up[j].prev)))
#                    break
#
#            if up[j].occupancy == parity:
#                p = self.priority(up[j].nid, self._domain.traverse_r(vertex, up[j].prev))
#                if slot is None or p <= slot[1]:
#                    slot = (j, p)
#            elif up[j].occupancy = Domain.Occupancy.EMPTY:
#                slot = (j, 0)
#
#        if slot is not None:
#            (s, p) = slot
#            evict = up[s].nid
#            q = self.priority(nid, self._domain.traverse_r(vertex, prev))
#
#            if (q, nid) > (p, evict):
#                up[s] = (nid, parity, prev)
#                self.put(vertex, Domain.pack(up[s]), s)
#                return (evict, s)
#
#        return None
# 
#    def resolve_provisional(self, vertex, slot, nid, prev):
#        """
#        Confirm neuron nid from vertex determined by prev has not been evicted from slot.
#
#        If nid and prev match, set slot occupamny to FULL and return True.
#        If nid or prev do not mach, return False.
#        """
#
#        u = Domain.unpack(self.get(vertex, slot))
#        if u.nid != nid or u.prev != prev:
#            return False
#        else:
#            u.occupancy = Domain.Occupancy.FULL
#            self.put(vertex, slot, Domain.pack(u))
#            return True
#


  
