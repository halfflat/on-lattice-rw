#
# 1. Parse arguments/configuration.
# 2. Select start locations, set initial conditions.
# 3. Run specified number of steps or until exhaustion:
#    For each tip:
#        a. Resolve stymied status.
#        b. Collect neighbour state.
#        c. Determine growth/branch behaviour.
#        d. Grow tip(s):
#             i.   If site has a vacancy, update tip state and fill vacancy;
#             ii.  Otherwise if site has provisional occupants, find lowest
#                  priority occupant and evict if lower priority than tip;
#             iii. Otherwise mark tip as stymied.
# 4. Final tidy-up: resolve any provisional tips.

# PRNG is backed by Random123 Philox4x64. The 128-bit key is composed from
# a 64-bit provided seed and a small integer denoting the function for
# which the generated numbers will be used.
#
# The NumPy Philox interface doesn't expose the Philox hashing function
# directly; we wrap the NumPy Philox bit generator in a class that munges
# the counter value in order to emulate this.

import sys
from collections import namedtuple
import numpy as np
import periodic_net as pn
from rw_domain import Domain, Occupancy

class Philox:
    """
    Wraps numpy.random.Philox bit generator.
    Provides method to set counter value explicitly.
    """

    # 'types' of function used in key construction
    GENERIC = 0
    INITIAL_SITE = 1
    PROPAGATION = 2
    PRIORITY = 3

    def __init__(self, seed, fn = GENERIC):
        """
        Create wrapped Philox generator with key derived from 64-bit seed and integer _fn_.
        """
        if seed < 0 or seed >= (1<<64):
            raise ValueError("seed must be representable as a 64-bit unsigned integer")
        key = (fn<<64)+seed

        self._bg = np.random.Philox(key=key)
        self._state0 = self._bg.state

        # The initial state0 has counter 0 and buffer_pos 4, so that the first call to random_raw will
        # bump the counter to 1 and return values from the hash of key and 1. Rewind counter to -1
        # so that we instead start with the hash of the key and 0.
        self.reset()

    def reset(self):
        np.copyto(self._state0.get('state').get('counter'), ~np.zeros((4,), dtype=np.uint64))
        self._bg.state = self._state0

    @property
    def bit_generator(self):
        return self._bg

    def set_counter(self, counter):
        if counter==0:
            self.reset()
        else:
            np.copyto(self._state0.get('state').get('counter'), np.frombuffer((int(counter)-1).to_bytes(32, sys.byteorder), dtype=np.uint64))
            self._bg.state = self._state0

    def __call__(self, counter = None):
        if counter is not None: self.set_counter(counter)
        return self._bg.random_raw()

# NumPy (based on a footnote in the documentation) appears to use Algoerithm 5
# of Lemire (2019) doi:10.1145/3230636. If we want identical results across Python
# and implementations in C++ or Fortran, we will need to ensure the same
# algorithm in each case.
#
# For scalar or array-like u, return integer or array of integers selected
# from half-open interval(s) [0, u) based on bits from BitGenerator bitgen.

def random_integer(bitgen, u):
    return np.random.Generator(bitgen).integers(u)

def mk_priority_fn(seed):
    """Construct a priority funciton f(nid, vtx_hash) for Domain."""
    hasher = Philox(seed, Philox.PRIORITY)
    def f(nid, vtx_hash):
        return hasher((nid<<64)+vtx_hash)
    return f

def mk_propagation_generator(seed):
    """Construct a funciton g(nid, vtx_hash, t) returning a BitGenerator."""
    p = Philox(seed, Philox.PROPAGATION)
    def g(nid, vtx_hash, t):
        p.set_counter((t<<96)+(nid<<72)+(vtx_hash<<8))
        return p.bit_generator
    return g

def mk_initial_random_vertex(domain, seed):
    hasher = Philox(seed, Philox.INITIAL_SITE)
    def f(nid, attempt):
        hasher.set_counter((nid<<64)+(attempt<<32))
        return random_integer(hasher.bit_generator, domain.extent)
    return f


class State:
    MAX_INITIAL_RETRY = 100

    # vertex is index tuple for domain; prev is the edge index from previous site to vertex;
    # slot is the slot in the site occupancy table at vertex where we expect to find this
    # tip unless it has been evicted.
    #
    # TODO: use a numpy compound datatype to save space?

    class Tip(namedtuple('Tip', ['vertex', 'prev', 'slot'])):
        def is_stymied(self):
            return self.prev==Domain.NOEDGE

        def stymied(vertex):
            return State.Tip(vertex=vertex, prev=Domain.NOEDGE, slot=None)

    def __init__(self, n_neurons, net, shape, max_occupancy, seed = 0):
        """
        Construct random walk state.

        Keyword arguments:
        n              -- number of neurons
        net            -- PeriodicNet object describing lattice
        shape          -- number (I, J) or (I, J, K) of lattice cells in domain
        max_occupancy  -- maximum number of neurons resident per site in domain
        """

        self._net = net
        self._domain = Domain(net, shape, max_occupancy, mk_priority_fn(seed))
        self._n = n_neurons
        self._t = 0
        self._multiple_occupancy = max_occupancy > 1
        self._propagation_generator = mk_propagation_generator(seed)

        self._tips = [[]] * self._n
        random_vertex = mk_initial_random_vertex(self._domain, seed)
        retry = []
        attempt = 0
        for i in range(0, self._n):
            nid = i + 1
            v = random_vertex(nid, attempt)
            match self._domain.occupy(v, nid, Occupancy.EVEN, Domain.NOEDGE):
                case (0, slot):
                    self._tips[nid-1] = [State.Tip(vertex=v, prev=Domain.NOEDGE, slot=slot)]
                case (evicted, slot):
                    self._tips[nid-1] = [State.Tip(vertex=v, prev=Domain.NOEDGE, slot=slot)]
                    self._tips[evicted-1] = []
                    retry.append(evicted)
                case None:
                    retry.append(nid)

        while retry:
            nid = retry[-1]
            del retry[-1]
            for attempt in range(1,State.MAX_INITIAL_RETRY+1):
                v = random_vertex(nid, attempt)
                print(v)
                match self._domain.occupy(v, nid, Occupancy.EVEN, Domain.NOEDGE):
                    case (0, slot):
                        self._tips[nid-1] = [State.Tip(vertex=v, prev=Domain.NOEDGE, slot=slot)]
                        break
                    case (evicted, slot):
                        self._tips[nid-1] = [State.Tip(vertex=v, prev=Domain.NOEDGE)]
                        self._tips[evicted-1] = []
                        retry.append(evicted)
                        break
                    case None:
                        continue
            else:
                raise Exception("unable to place neuron {}".format(nid))

        # Keep a copy of initial neuron sites so that neuron morphology can be reconstructed.

        self.initial = np.empty((nid, len(self._domain.extent)), dtype=np.uint32)
        for i, tip1 in enumerate(self._tips):
            self.initial[i,:] = np.asarray(tip1[0].vertex, dtype=np.uint32)

    domain = property(lambda self: self._domain)

    def step(self, final = False):
        self._t += 1

        n_exhausted = 0
        for i in range(0, self._n):
            nid = i + 1
            ts = self._tips[i]

            # Work backwards through list of tips: new tips from branching are appended;
            # tips that expire are replaced by the tip at the end, and the end is then removed.
            for j in range(len(ts)-1, -1, -1):
                tip = ts[j]

                # Check if tip was evicted, fix occupancy state.
                if not tip.is_stymied():
                    if not self._domain.resolve_provisional(tip.vertex, tip.slot, nid, tip.prev):
                        assert tip.prev != Domain.NOEDGE
                        ts[j] = State.Tip.stymied(self._domain.traverse_r(tip.vertex, tip.prev))

                if final: continue
                if self._multiple_occupancy: self.step_mo(self._t, nid, ts, j)
                else:                        self.step_so(self._t, nid, ts, j)

            if not self._tips[i]: n_exhausted += 1
        return n_exhausted != self._n

    def step_mo(self, t, nid, ts, j):
        # mo > 1 case:
        # 1. Branch if tip is not stymied and site has shared occupancy.
        # 2. Select and propagte to new sites from neighbours which are (provisionally)
        #    not full and which are not already occupied by same nid.

        parity = Occupancy.parity(t)
        tip = ts[j]
        branch = not tip.stymied() and self._domain.has_other_nid(tip.vertex, nid, parity)
        neighbours = self._domain.open_neighbours_excluding(tip.vertex, nid, parity)

        if not neighbours:
            # just remove this tip
            ts[j] = ts[-1]
            del ts[-1]
            return

        bg = None
        if len(neighbours)==1:
            k = 0
        else:
            bg = self._propagation_generator(nid, self._domain.vertex_hash(tip.vertex), t)
            k = random_integer(bg, len(neighbours))

        (u, e) = neighbours[k]
        match self._domain.occupy(u, nid, parity, e):
            case (_, s): ts[j] = State.Tip(vertex=u, prev=e, slot=s)
            case None:   ts[j] = State.Tip.stymied(tip.vertex)

        if not branch or len(neighbours)<2:
            return

        neighbours[k] = neighbours[-1]
        del neighbours[-1]
        if len(neighbours)==1:
            k = 0
        else:
            k = random_integer(bg, len(neighbours))

        (u, e) = neighbours[k]
        match self._domain.occupy(u, nid, parity, e):
            case (_, s): ts.append(State.Tip(vertex=u, prev=e, slot=s))
            case None:   ts.append(State.Tip.stymied(tip.vertex))

    def step_so(self, t, nid, ts, j):
        # mo == 1 case:
        # 1. Select new site from neighbours which are not already occupied
        #    by same nid.
        # 2. If new site is full, branch and:
        #    3. Select and propagate to new sites from neighbours
        #       which are (provisionally) not full and not already occupied by
        #       same nid.
        #    Otherwise:
        #    3. Propagate to new site.

        parity = Occupancy.parity(t)
        tip = ts[j]
        neighbours = self._domain.neighbours_excluding(tip.vertex, nid, parity)

        if not neighbours:
            # just remove this tip
            ts[j] = ts[-1]
            del ts[-1]
            return

        bg = self._propagation_generator(nid, self._domain.vertex_hash(tip.vertex), t)
        k = random_integer(bg, len(neighbours))
        (u, e) = neighbours[k]

        if not self._domain.is_full(u, parity):
            match self._domain.occupy(u, nid, parity, e):
                case (_, s): ts[j] = State.Tip(vertex=u, prev=e, slot=s)
                case None:   ts[j] = State.Tip.stymied(tip.vertex)
        else:
            # Brannch: select two destinations from open neighbours.
            neighbours = self._domain.open_neighbours_excluding(tip.vertex, nid, parity)
            if not neighbours:
                # remove tip and return
                ts[j] = ts[-1]
                del ts[-1]
                return

            k = random_integer(bg, len(neighbours))
            (u, e) = neighbours[k]
            match self._domain.occupy(u, nid, parity, e):
                case (_, s): ts[j] = State.Tip(vertex=u, prev=e, slot=s)
                case None:   ts[j] = State.Tip.stymied(tip.vertex)

            if len(neighbours) == 1:
                return

            neighbours[k] = neighbours[-1]
            k = random_integer(bg, len(neighbours)-1)
            (u, e) = neighbours[k]
            match self._domain.occupy(u, nid, parity, e):
                case (_, s): ts[j] = State.Tip(vertex=u, prev=e, slot=s)
                case None:   ts[j] = State.Tip.stymied(tip.vertex)

