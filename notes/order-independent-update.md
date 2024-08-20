# Order indepdent update

To avoid bias in any given instance of a random walk, the individual points
of propagation ('tips') can be grown in an arbitrary, random order, where
tips propagated earlier take precedence over those propagated later.

This can be achieved by randomly permuting the sequence of tips each time
step but the order dependence limits the possibilities of parallelism:
multiple threads of execution would need to ensure that the locations of
the tips for which they are responsible were sufficiently isolated so
that their growth could be computed independently; this poses an even thornier
issue for decomposition in the distributed context.

Under the assumption that a neuron can occupy any site at most once,
an alternative approach is to provide an ordering function _D(n, x)_ for
neurons _n_ and sites _x_ such that at some time step, a potential growth
into the same site _w_ of two tips corresponding to neurons _n_ and _m_ at sites
_x_ and _y_ respectively will prioritize neuron _n_ over _m_ if there is insufficient
occupancy to admit both if _D(n, x)_ &gt; _D(m, y)_.

The function _D_ is selected from a distribution such that each value _D(n, x)_
is i.i.d. uniformly over some non-empty real interval. In practice, this
can be simulated by a counter-based random number generator [@SMDS2011]
keyed on a hash of _n_ and _x_. In the very unlikely advent of a collision
_D(n, x)_ = _D(m, y)_, the tie can be resolved based on another CRNG evaluation keyed by
a hash of _n_, _m_, _x_, and _y_ or with presumably negligable bias, simply
by comparing _n_ and _m_.

The advantage of using a total order across all tips is that in the case
of potential multiple occupancy of a site, a minimum priority for the site
can be maintained which allows early determnination of admissibility.

## Serial update process

The working set comprises the domain grid taking as values an array of _P_
occupancy slots and a set of propagating tips _T_.

Each occupancy slot (_n_, _x_; _F_) has three fields: a neuron id _n_, a
previous site _x_ (represented by an edge index of the periodic net), and a
provisional flag _F_. The flag _F_ can take one of four values:

* 'empty', indicating that this slot is unoccupied;

* 'odd', indicating that this slot is provisionally occupied after a
  time step _t_ with integer _t_ odd.

* 'even', indicating that this slot is provisionally occupied after a
  time step _t_ with integer _t_ even.

* 'full', indicating that this slot is unprovisionally occupied.

At time step _t_, a slot is regarded as occupied if
its flag is **full** or it is provisionally occupied with a flag
that has the opposite parity to that of time _t_. (That is, if _t_
is for example even, a flag value of **full** or **odd** implies
the slot is occupied.)

A tip _p_ = (_n_, _x_, _f_, _S_) comprises:

* a neuron id _n_,

* a site (vertex) _x_,

* a former site _f_ (this can be represented by an edge index from _x_),

* and a 'stymied' flag _S_.

The former site value _f_ is ignored if the tip is stymied; the former
site is in this circumstance regarded as being the same as the site _x_.

Marking _p_ as stymied sets _x_ to _f_ and _S_ to **true**.

At initialization:

1. Initialize the grid by filling each occupancy slot at each site
   with (0, 0; **empty**); the neuron id 0 is reserved to indicate
   the absense of a neuron.

2. For each neuron with id _n_ select its starting grid location _x₀_(_n_)
   and create a corresponding 'tip' _p_ for neuron _n_ at site _x₀_ and
   unstymied (i.e. _S_ is false). Fill the first occupancy slot at _x₀_
   with (_n_, _?_; **full**), where _?_ is arbitrary.

For each time step _t_, let _P_ be the parity of _t_ and _Q_ the opposite
parity, and then consider each tip _p_ = (_n_, _x_, _f_, _S_):

* If _S_ is false, determine if the tip was ejected by a different tip:
  is there a slot at _x_ with (_n_, _f_; _Q_)? If so, propagation was
  successful and the slot is updated to (_n_, _f_; **full**). Otherwise
  the tip was ejected and marked as stymied (i.e. _x_ is set to _f_
  and _S_ is set to **true** for _p_).

* If _S_ is false, determine if the tip branches based on the occupation
  status of _x_ and its neighbouts.

* Consider the set of sites neighbouring _x_ which are not completely occupied;
  if it empty, discard tip _t_. Otherwise, randomly select a site _w_ or two
  distinct sites _w_ and _w'_ if the tip branches and there are two available
  such sites. Update tip _p_ ← (_n_, _w_, _x_, **false**); if there is
  a _w'_, also create a new tip _p'_ = ( _n_, _w'_, _x_, **false**).

* For the tip _p_, perfom the _update operation_ at _w_:

    1. If there is a slot (_n_, _z_; _P_) with the same neuron id _n_:
       if _D(n, x)_ &lt; _D(n, z)_, mark _p_ as stymied (see above); otherwise,
       replace the slot with (_n_, _x_; _P_).

    2. Otherwise find the provisional entry (_m_, _z_; _P_) with lowest priority
       as determined by _D_ (empty slots always have lowest priority) and
       replace that slot with (_n_, _x_; _P_).

* Perform the same update operation for _p'_ at _w'_.

## Parallel update operation

The serial update operation can be applied if there is a mutex associated with
each site, and site data is then updated while holding the lock. Alternatively,
this can also be done lock-free with atomics updates.

For the single occupancy case, this is straightforward:

1. The current occupier (_m_, _y_; _F_), if any, is retrieved.

2. If the tip has lower priority than the current occupier, mark the tip as stymied and continue.

3. Otherwise, perform a (strong) atomic compare and swap replacing the representation
   of (_m_, _y_; _F_) with that of (_n_, _x_; _P_); on failure, retry from step 1.

For multiple occupancy, a lock-free approach is more involved. The following
approach is predicated upon the assumption that the maximum occupancy is
still fairly small, e.g. on the order of tens of neurons, not thousands.

The site occupancy data is augmented with a key describing the lowest
priority provisional occupier (_m_, _y_; _F_, _i_) where _i_ is the index into the
occupancy array. Each entry in the occupancy array is also extended with the
array index so that we can perform a bit-wise comparison between the the key
and the corresponding occupancy entry.

The site update process is then:

1. Retrieve the lowest priority occupier key _K_ = (_m_, _y_; _F_, _i_) as above.

2. If _D(n, x)_ &lt; _D(m, y)_, mark the tip as stymied and continue.

3. Atomic compare-and-swap the ith occupancy slot, replacing _K_ with _K'_ =
    (_n_, _x_; _F_, _i_). On failure, retry from step 1.

4. Scan the other provisional entries in the occupancy slots:

    1. If there is an entry (_n_, _z_; _F_, _j_) with the same neuron id _n_:
       if _D(n, x)_ &lt; _D(n, z)_, restore the value _K_ in slot _i_, mark the
       tip as stymied, and continue; otherwise write (_n_, _x_; _F_, _j_) to
       slot _j_, then restore _K_ to slot _i_.

    2. Otherwise find the provisional entry with lowest priority as determined
       by _D_ and copy its value to the site's lowest priority occupier key.

Empty, available slots are represented by special values of _m_, _y_ and _F_ in
the occupancy slots and key, but retain the index _i_. They are regarded as
having minimum priority for the purposes of comparison.

This site update process (should) maintain exclusive access for manipulation of
the occupancy slots: potential writers must wait until the lowest priority
occupier key _K_ has the same value as the referenced occupier slot; modifying
the slot then excludes access until the key is updated. Further, the minimum
priority _D_ can only increase monotonically. Stores (or CAS success) and
loads (or CAS failure) should admit relaxed memory order release and
acquire semantics respectively.


## Distributed parallel update process

TODO: expand from notes below

* Neighbourhood relationships are greatly simplified if we split the domains
  only across one (e.g. z) axis and use a 1-cell halo rather than the strict
  1-neighbourhood of the subdomain (for e.g. the periodic cubic net these
  will be the same anyway).

* Despite what I thought earlier, a 1-cell halo should suffice with the
  following protocol:

* Each rank is responsible for propagating only the tips that it 'owns',
  which are exactly those whose most recent location is in their exclusive
  subdomain (and not the halo).

* After performing local propagation, perform a neighbour exchange: to
  each neighbour send the set of updated site data for each site in
  the subdomain ∩ neighbour's halo _and_ any tips that have locally
  been determined to have (provisionally) crossed into the neighbour's
  domain.

* After exchange, for each tip received from neighbour check if provisional
  propagation into domain succeeded; if and only if so, add tip to local
  tip set and update site. Similarly, for each tip send to neighbour, check
  if propagation into halo succeeded; if and only if so, update site in
  halo and remove tip from local tip set, otherwise mark tip as stymied.

## Occupancy representation

Without resorting to more exotic compression schemes, storing a neuron id of
maximum value _N_, source information _x_ and a provisional flag value _F_ can
be packed into ⌈log₂ _N_⌉ + ⌈log₂ _C_⌉ + 2 bits (for _F_), where _C_ is the
coordination number (or degree) of the periodic net, as _x_ is represented by
the edge index spanned by the most recent extension of the neuron into the
site. The maximum value of _C_ is 12 for the **fcn** net [see @DFOY2003a, Table 1],
requiring 4 bits, but all other regular and semi-regular nets have _C_ ≤ 8.

For single occupancy representations then, and reserving two ids to represent
unoccupied and margin sites, 2^26-2 = 67'108'862 distinct neurons can be
admitted while using a single 32-bit word (4 bytes) per site.

For multiple occupancy of size _P_ using the lock-free scheme described above,
an additional ⌈log₂ _P_⌉ bits are required per slot together with an additional
requirement for the lowest-priority-occupier-key. Using 32-bit words the
storage requirement is _P_ + 1 words (4 _P_ + 4 bytes) per site with a maximum
of 2^(26 - ⌈log₂ _P_⌉) - 2 distinct neurons. For example, with _P_ = 4, maximum
_N_ would be 16'777'214 (reserving the all zero bits and all one bits
representations).

Doubling the per-site storage through using a 64-bit word would, however, allow
a ridiculously large number of distinct neurons, even in the face of moderately
large _P_.

An implementation could determine the appropriate word size and packing scheme
at program run time.


---
references:
- type: paper-conference
  id: SMDS2011
  author:
  - family: Salmon
    given: John K.
  - family: Moraes
    given: Mark A.
  - family: Dror
    given: Ron O.
  - family: Shaw
    given: David E.
  issued:
    year: 2011
  title: 'Parallel random numbers: as easy as 1, 2, 3'
  container-title: 'Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis'
  publisher: 'Association for Computing Machinery'
  number: 16
  DOI: 10.1145/2063384.2063405
...

