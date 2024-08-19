# Array-based neuron lattice random walk implementation

This implementation substitutes the TreeNode representation of the periodic
graph with an array-based representation.

The periodic graphs are regular (each vertex has the same degree) and should
have an embedding in 2- or 3-dimensional Euclidean space with constant edge
lengths and maximal symmetry about each vertex. The periodicity corresponds to
invariance under translation by (respectively) 2 or 3 independent vectors.
These graphs broadly correspond to _crystal nets_ — following the conventions
described in [@DFO2005], they are n-regular 2- or 3-periodic nets. To avoid
confusion of terms in the literature, the authors use the term _coordination_
to refer to vertex degree.

The invariant translations are not unique. The fundamental domain under the
action of the translations in Euclidean space is a _unit cell_; a domain
with minimal volume is termed a _primitive unit cell_. A larger 'conventional'
unit cell can be chosen for convenience, for example, with a set of orthogonal
translations.

The translation vectors generate a free Abelian group ℤⁿ that acts freely
on the graph. One can then form a quotient graph under this action; edges
in the periodic graph that connect a vertex _u_ in the unit cell to a
vertex _v_ in the unit cell translated by an element _g_ of the group
constitute a single edge in the quotient graph between _[u]_ and _[v]_;
this is then labelled as a pair of directed edges, label _g_ from
_[u]_ to _[v]_ and label _g⁻¹_ from _[v]_ to _[u]_.

## Representation of periodic nets

A regular 2- or 3-periodic net can be represented by the following

* The order of the translations, or equivalently the dimension of the
  embedding: 2 for a planar embedding, 3 for a spatial embedding.

* The number of vertices in the quotient graph.

* The 2 or 3 translation vectors that determine the unit cell
  in 2- or 3-dimensional Euclidean space respectively.

* For each vertex in the quotient, the list of edges. Each edge
  is represented by the index of the neighbouring vertex and the
  outgoing edge label in the quotient as a tuple (a, b) or (a, b, c)
  corresponding to the element of the 2- or 3-rank group generated
  by the translations.

* The inverse edge map I: if the outgoing edge with index _i_ from vertex _u_
  has neighbour _v_ and label _g_, then the outgoing edge with index _I(i)_
  of _v_ has neighbout _u_ and label _g⁻¹_.

* For each vertex in the quotient, the offset into the unit cell as a
  vector in the Euclidean space.

## Representation of the domain

For a 3-periodic net, the bounded domain is represented by an array
indexed with _i_, _j_, _k_, and _r_, representing the vertex with index
_r_ in the unit cell translated by the vector _ia_ + _jb_ + _kc_, given
translation vectors _a_, _b_, _c_.

In order to simplify bounds checking, the domain of I×J×K unit cells
each with N vertices corresponds to array indices (1..I,1..J,1..K,0..N-1);
the full array has dimensions (I+2,J+2,K+2,N) and elements with indices
_i_=0 or I+1, (or _j_=0 or J+1, or _k_=0 or K+1) are given special
values indicating that are points in the margin of the domain.

The value type in the array will depend upon the details of the random
walk implementation, but at minimum will represent the identifiers of
any occupying neurons. If neuron geometry is to be recovered from the
values of the domain array, the values will also need to capture the
edge index from which the occupying neuron came, if any.

---

references:
- type: article-journal
  id: DFOY2003a
  author:
  - family: Delgado-Friedrichs
    given: Olaf
  - family: O'Keeffe
    given: Michael
  - family: Yaghi
    given: Omar M.
  issued:
    year: 2003
  title: 'Three-periodic nets and tilings: regular and quasiregular nets'
  container-title: 'Acta Crystallographica Section A'
  volume: 59
  page: 22-27
  DOI: 10.1107/S0108767302018494
- type: article-journal
  id: DFOY2003b
  author:
  - family: Delgado Friedrichs
    given: Olaf
  - family: O'Keeffe
    given: Michael
  - family: Yaghi
    given: Omar M.
  issued:
    year: 2003
  title: 'Three-periodic nets and tilings: semi-regular nets'
  container-title: 'Acta Crystallographica Section A'
  volume: 59
  page: 515-525
  DOI: 10.1107/S0108767303017100
- type: article-journal
  id: DFO2005
  author:
  - family: Delgado-Friedrichs
    given: Olaf
  - family: O'Keeffe
    given: Michael
  issued:
    year: 2005
  title: 'Crystal nets as graphs: Terminology and definitions'
  container-title: 'Journal of Solid State Chemistry'
  volume: 178
  issue: 8
  page: 2480–2485
  DOI: 10.1016/j.jssc.2005.06.011
...
