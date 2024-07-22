import numpy as np
import random

# Notation and terms:
#
# Lattice points and edges form a n-dimensional net, that is, a graph
# which has an embedding in n-dimensional Euclidean space with
# translational symmetry by n independent vectors in that space.
# These translations in turn determine a corresponding free group
# action of ℤⁿ on the graph.
#
# Vertices in the lattice are identified with vertices in the
# quotient graph under the group action, and edges between vertice
# with the edges in the quotient, labelled by the translations.
#
# One vertex in the lattice then is given by an index into the
# vertex set of the quotient and a unit cell coordinate given
# by an element of ℤⁿ. Edges from a lattice point are given
# by a unit cell displacement in { (xᵢ)∈ℤⁿ | |xᵢ| ≤ 1 ∀i}
# and the index of its neighbour.
#
# When n=3 these points and edges can be represented in three
# dimensional Euclidean space. If the translations are axis-aligned,
# it is then also straightforward to represent the lattice points
# within an axis-aligned bounding box.


# Lattice comprises:
#   - order of translations: 2 => planar lattice, 3 => space lattice
#   - number of lattice points in the unit cell/quotient graph
#   - edge list for each point in the unit cell
#   - inverse edge map I: if the edge from p with index i takes p to q
#     then the edge from q with index I(i) takes q back to p.
#   - the translations as vectors in 2- or 3-d Euclidean space.
#   - the offset into a unit cell for each vertex as a vector in
#     2- or 3-d Euclidean space

# It is assumed the lattices correspond to regular graphs.

class LatticeConfiguration:
    def __init__(self, edgemap, positions, translations):
        # edgemap is an integer numpy array with one row per edge.
        # Each row has the format:
        #     [ vertex-index x1 x2 ... xn neighbour-index]
        # where the vertex indices are 0-based and x1, x2, ...
        # are the translations to the corresponding neighbouring
        # unit cell. Edges are in order of vertex-index.

        self.edgemap = edgemap.copy()
        self.positions = translations.copy()
        self.translations = translations.copy()

        self.n_vertices = edgemap[:,0].max()+1
        self.degree =  np.uint32(edgemap.shape[0]/self.n_vertices)
        self.dimension = edgemap.shape[1]-2

        if (positions.shape != (self.n_vertices, self.dimension)):
            raise Exception("Bad position array")

        if (translations.shape != (self.dimension, self.dimension)):
            raise Exception("Bad position array")


        # compute inverse edge map from vertex 0; then verify
        # for other points.

        n_edges = edgemap.shape[0]
        self.inv_edges = np.empty((n_edges,),dtype=np.int32)
        self.inv_edges.fill(-1)

        for r in range(0, n_edges):
            ni = edgemap[r,1]
            ie = edgemap[r,:].copy()
            ie[[0,-1]] = ie[[-1,0]]
            ie[1:(1+self.dimension)] *= -1
            # neighbour vertex is ie[0]; search corresponding span of edgemap
            for s in range(ie[0]*self.degree, (ie[0]+1)*self.degree):
                if np.array_equal(ie, edgemap[s,:]):
                    self.inv_edges[r] = s
                    break
            if self.inv_edges[r]==-1:
                raise Exception("couldn't find inverse edge")



class ArrayLattice:
    border_value = np.invert(np.uint32(0))

    def __init__(self, lattice_config, shape):
        # for now? just have one item per lattice point, viz. neuron index + parent edge index
        # packed into 32 bytes.

        self.config = lattice_config
        if self.config.dimension != len(shape):
            raise Exception("shape is wrong dimension")

        # make lattice but fill boundary with boundary marker
        inner = shape + (self.config.n_vertices,)
        pad = [(1,1)]*len(inner)
        pad[-1] = (0,0)

        self.pad_offset = np.ones((len(inner),), dtype=np.int32)
        self.pad_offset[-1] = 0

        self.lattice = np.pad(np.zeros(inner, dtype=np.uint32), pad, mode='constant', constant_values=ArrayLattice.border_value)

    # return tuple (neuron id, from edge); cell indices are one-based; vertex index (last component) is zero-based.
    def get(self, vertex):
        packed = self.lattice[tuple(vertex)]
        return (np.right_shift(packed, 8), np.bitwise_and(packed, 255))

    def put(self, vertex, nid, from_edge):
        packed = np.left_shift(np.uint32(nid), 8) + np.uint8(from_edge)
        self.lattice[tuple(vertex)] = packed

    # return tuple (occupancy, n_free). occupancy is an array of shape (degree, 2);
    # each row is [edge-index, number-of-neurons] with -1 indicating boundary padding.
    # n_free is the number of neighbours with 0 neurons.

    # (TODO probably better to return packed neuron list (current of length 1) instead of
    # number of neurons).

    def neighbours(self, vertex):
        vi = vertex[-1]
        degree = self.config.degree
        occupancy = np.empty((degree, 2), np.int32)
        n_free = 0
        for i in range(0, degree):
            edge_idx = i+vi*self.config.degree
            neighbour = vertex + self.config.edgemap[edge_idx,1:]
            packed = self.lattice[tuple(neighbour)]
            occupancy[i, 0] = edge_idx
            if packed == ArrayLattice.border_value:
                occupancy[i, 1] = -1
            elif packed == 0:
                occupancy[i, 1] = 0
                n_free += 1
            else:
                occupancy[i, 1] = 1
        return occupancy, n_free

    def traverse(self, vertex, edge_index):
        return vertex + self.config.edgemap[edge_index,1:]

    def position(self, vertex):
        unpad = vertex - self.pad_offset
        return self.config.positions[unpad[-1]] + np.matmul(unpad[:self.config.dimension], self.config.translations)

# boring 2-d grid for starters!
#
# Only one vertex in cell, four edges to neighbouring cells.

square_lattice = LatticeConfiguration(
        np.array([[0, 1, 0, 0],[0, 0, 1, 0],[0, -1, 0, 0],[0, 0, -1, 0]]),
        np.array([[0.5, 0.5]]), # pop vertex in middle of unit cell
        np.array([[1, 0], [0, 1]])) # map translations into vectors on plane.


lattice = ArrayLattice(square_lattice, (4,4))

# basic self-avoiding random walk on lattice

start = (1,1,0)
lattice.put(start, 1, 0)

def rand_free_edge(occupancy, n_free):
    degree = occupancy.shape[0]
    k = random.randrange(n_free)
    for i in range(0, degree):
        if occupancy[i,1]==0:
            if k==0:
                return i
            else:
                k -= 1
    return -1

v = start
neuron_id = 1
while True:
    occupancy, n_free = lattice.neighbours(v)
    #print(occupancy)
    #print(n_free)

    if n_free==0:
        break

    edge = rand_free_edge(occupancy, n_free)
    if edge==-1:
        raise Exception("oops")

    v = lattice.traverse(v, edge)
    lattice.put(v, neuron_id, edge)

# follow walk from start and output coordinates
print(lattice.lattice)

v = start
while True:
    print(lattice.position(v))

    occupancy, n_free = lattice.neighbours(v)
    next_vertex = None
    for edge in range(0, occupancy.shape[0]):
        if occupancy[edge, 1]>0:
            # is it us, and did it come from this vertex?
            w = lattice.traverse(v, edge)
            nid, from_edge = lattice.get(w)
            if nid==neuron_id and edge==from_edge:
                next_vertex = w
                break
    if next_vertex is None:
        break
    else:
        v = next_vertex

