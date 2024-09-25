import enum
import math
import numbers
import numpy as np

class Occupancy(enum.IntEnum):
    EMPTY = 0
    ODD = 1
    EVEN = 2
    FULL = 3

    def flip(self):
        match self:
            case Occupancy.EVEN: return Occupancy.ODD
            case Occupancy.ODD:  return Occupancy.EVEN
            case _: return self

    def parity(t):
        return Occupancy.ODD if t%2 else Occupancy.EVEN

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
    # describing each occupant: the neuron id (24 bits); occupancy class
    # (2 bits); and previous edge index (6 bits).

    BORDER = np.invert(np.uint32(0))
    NOEDGE = np.uint8(63)
    OCCUPANT_DTYPE = np.dtype([('nid', np.uint32), ('occupancy', np.uint8), ('prev', np.uint8)])

    def __init__(self, net, shape, max_occupancy, priority):
        """
        Create and initialize a domain for lattice random walk

        Keyword arguments
        net              -- a PeriodicNet object describing the lattice
        shape            -- tuple (I, J) or (I, J, K) describing extent of
                            domain in lattice cells
        max_occupancy    -- maximum number of residents per lattice site
        priority         -- function p(nid, vtx_hash) that returns the
                            numerical priority of neuron nid coming from the
                            vertex represented by the integer vtx_hash
        """

        self._mo = max_occupancy;
        self._net = net
        self._priority_fn = priority

        if self._net.dimension != len(shape):
            raise Exception("shape is wrong dimension")

        if self._net.degree>63:
            raise Exception("only supporting up to degree 63 with packed representation")

        # make domain but fill boundary with boundary marker
        inner = shape + (self._net.n_vertices, self._mo)
        pad = [(1,1)]*len(inner)
        pad[-2:] = [(0,0), (0,0)]

        self._pad_offset = np.ones((len(inner),), dtype=np.int32)
        self._pad_offset[-2:] = [0, 0]

        self._array = np.pad(np.zeros(inner, dtype=np.uint32), pad, mode='constant', constant_values=Domain.BORDER)

        # pre-calculate offsets, shifts for computing vertex index hashes
        self._vext = self._array.shape[:-1]
        self._vdim = len(self._vext)
        self._hw = np.asarray([0]+[int(k-1).bit_length() for k in self._vext])
        self._ho = self._hw.cumsum()

    # Return extent, excluding the padded boundaries and occupancy axis.
    extent = property(lambda self: (self._array.shape - self._pad_offset*2)[:-1])

    # Use vertex with coordinates matching extent to indicate a dummy vertex
    no_vertex = property(lambda self: self.extent)

    # Return number of occupancy slots per vertex
    max_occupancy = property(lambda self: self._mo)

    # Access to underlying net
    net = property(lambda self: self._net)

    def priority(self, nid, vertex_or_hash):
        if isinstance(vertex_or_hash, numbers.Integral):
            return self._priority_fn(nid, vertex_or_hash)
        else:
            return self._priority_fn(nid, self.vertex_hash(vertex_or_hash))

    def unpack(packed):
        u = np.recarray(packed.shape, dtype=Domain.OCCUPANT_DTYPE)
        u.nid = np.right_shift(packed, 8)
        u.occupancy = np.right_shift(np.bitwise_and(packed, 255), 6)
        u.prev = np.bitwise_and(packed, 63)
        return u

    def pack(unpacked):
        return np.array(np.left_shift(unpacked.nid, 8) + np.left_shift(unpacked.occupancy, 6) + unpacked.prev, dtype=np.uint32)

    # Convert vertex index to and from packed bit representation
    def vertex_hash(self, vertex):
        return (np.asarray(vertex) << self._ho[:-1]).sum()

    def vertex_unhash(self, h):
        return (h >> self._ho[:-1]) & ((1 << self._hw[1:]) - 1)

    vertex_hash_bits = property(lambda self: self._ho[-1])

    # Return occupant in given slot in the domain at vertex, or all occupants at vertex if slot is None.
    def get(self, vertex, slot = None):
        index = np.asarray(vertex) + self._pad_offset[:-1]
        if slot is None:
            return self._array[tuple(index)+(slice(None),)]
        else:
            return self._array[tuple(index)+(slot,)]

    # Update occupant at slot in the domain at vertex with packed value, or all occupants with array of packed values if slot is None.
    def put(self, vertex, occupant, slot = None):
        index = np.asarray(vertex) + self._pad_offset[:-1]
        if slot is None:
            self._array[tuple(index)+(slice(None),)] = occupant
        else:
            self._array[tuple(index)+(slot,)] = occupant

    def traverse(self, vertex, edge_index):
        if edge_index == Domain.NOEDGE: return self.no_vertex
        return self._net.traverse(vertex, edge_index)

    def traverse_r(self, vertex, edge_index):
        if edge_index == Domain.NOEDGE: return self.no_vertex
        return self._net.traverse_r(vertex, edge_index)

    def position(self, vertex):
        return self._net.position(vertex)

    def is_border(self, vertex):
        index = np.asarray(vertex) + self._pad_offset[:-1]
        return self.get(vertex, 0)==self.BORDER

    def has_nid(self, vertex, nid, parity):
        """
        Test if site at vertex holds nid unprovisionally.

        Return True if any slot has matching nid with occupancy FULL or opposite of _parity_.
        """
        up = Domain.unpack(self.get(vertex))
        return ((up.nid == nid) & ((up.occupancy == Occupancy.FULL) | (up.occupancy == parity.flip()))).any()

    def has_other_nid(self, vertex, nid, parity):
        """
        Test if site at vertex holds a different nid unprovisionally.

        Return True if any slot has a different nid with occupancy FULL or opposite of _parity_.
        """
        up = Domain.unpack(self.get(vertex))
        return ((up.nid != nid) & ((up.occupancy == Occupancy.FULL) | (up.occupancy == parity.flip()))).any()

    def is_full(self, vertex, parity):
        """
        Test if site at vertex may accomodate another neuron.

        Returns True if every slot has occupancy FULL or opposite of _parity_.
        """
        up = Domain.unpack(self.get(vertex))
        return ((up.occupancy == Occupancy.FULL) | (up.occupancy == parity.flip())).all()

    def neighbours(self, vertex):
        """Non-border neighbours [(u,e)] of vertex with u = traverse(vertex, e) which do not hold nid"""
        return [(u, e) for e in self._net.edge_indices_from(vertex[-1]) for u in [self.traverse(vertex, e)] if not self.is_border(u)]

    def neighbours_excluding(self, vertex, nid, parity):
        """Non-border neighbours [(u,e)] of vertex with u = traverse(vertex, e)"""
        return [(u, e) for e in self._net.edge_indices_from(vertex[-1]) for u in [self.traverse(vertex, e)] if not self.is_border(u)
                and not self.has_nid(u, nid, parity)]

    def open_neighbours_excluding(self, vertex, nid, parity):
        """Non-border neighbours [(u,e)] of vertex which are not full and do
           not hold nid with u = traverse(vertex, e)."""
        return [(u, e) for e in self._net.edge_indices_from(vertex[-1]) for u in [self.traverse(vertex, e)] if not self.is_border(u)
                and not self.is_full(u, parity) and not self.has_nid(u, nid, parity)]

    def occupy(self, vertex, nid, parity, prev):
        """
        Provisionally place (nid, parity, prev) in a slot at vertex.

        A placement will succeed if there is an EMPTY slot or a provisional
        slot, viz. a slot with the same occupancy _parity_, with lower
        priority. A vertex can host at most one instance of a neuron nid so
        if nid is already present, placement can only succeed if that instance
        is provisional and of lower priority.

        Returns None on failure, or else a pair (evicted, slot-index) where
        evicted is the id of a neuron that was removed to make room for this
        neuron (or zero, if the slot was empty) and slot-index indicates which
        occupancy slot was occupied.
        """

        up = Domain.unpack(self.get(vertex))
        slot = None   # or (slot index, slot priority)

        for j in range(self._mo):
            if up[j].nid == nid:
                if up[j].occupancy != parity:
                    return None
                else:
                    slot = (j, self.priority(up[j].nid, self.traverse_r(vertex, up[j].prev)))
                    break

            if up[j].occupancy == parity:
                p = self.priority(up[j].nid, self.traverse_r(vertex, up[j].prev))
                if slot is None or p <= slot[1]:
                    slot = (j, p)
            elif up[j].occupancy == Occupancy.EMPTY:
                match slot:
                    case (_, 0): pass
                    case _: slot = (j, 0)

        if slot is not None:
            (s, p) = slot
            evict = up[s].nid
            q = self.priority(nid, self.traverse_r(vertex, prev))

            if (q, nid) > (p, evict):
                up[s] = (nid, parity, prev)
                self.put(vertex, Domain.pack(up[s]), s)
                return (evict, s)

        return None

    def resolve_provisional(self, vertex, slot, nid, prev):
        """
        Confirm neuron nid from vertex determined by prev has not been evicted from slot.

        If nid and prev match, set slot occupamny to FULL and return True.
        If nid or prev do not mach, return False.
        """

        u = Domain.unpack(self.get(vertex, slot))
        if u.nid != nid or u.prev != prev:
            return False
        else:
            u.occupancy = Occupancy.FULL
            self.put(vertex, Domain.pack(u), slot)
            return True
