#
# 1. Parse arguments/configuration.
# 2. Select start locations, set initial conditions.
# 3. Run specified number of steps or until exhaustion:
#    For each tip:
#        a. Resolve stymied status.
#        b. Collect neighbor state.
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

import numpy as np
import enum
import sys
from periodic_net import Domain

class Philox:
    # 'types' of function used in key construction
    GENERIC = 0
    INITIAL_SITE = 1
    PROPAGATION = 2
    PRIORITY = 3

    def __init__(self, seed, fn = self.GENERIC)
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

    def bit_generator(self):
        return self._bg

    def set_counter(self, counter):
        if counter==0:
            self.reset()
        else:
            np.copyto(self._state0.get('state').get('counter'), np.frombuffer((counter-1).to_bytes(32, sys.byteorder), dtype=np.uint64))
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

def mk_priority_map(domain, seed):
    if domain.vertex_hash_bits>64:
        raise ValueError("domain extent is unexpectedly large")

    hasher = Philox(seed, Philox.PRIORITY)
    def f(nid, vertex_or_hash):
        if type(vertex_or_hash) is int:
            return hasher((nid<<64)+vertex_or_hash)
        else:
            return hasher((nid<<64)+domain.vertex_hash(vertex_or_hash))
    return f

def mk_initial_map(domain, seed):
    hasher = Philox(seed, Philox.INITIAL_SITE)
    def f(nid, attempt):
        hasher.set_counter((nid<<64)+(attempt<,32))
        return random_integer(hasher, domain.extent())
    return f

class State:
    class Tip(namedtuple('Tip', ['vertex', 'from_edge']):
            stymied(self): return self.from_edge<0
            stymy(self): self.from_edge = -1

    def __init__(domain, n_neurons, seed = 0)
        self._domain = domain
        self._cbrng = Philox(seed)
        self._n = n_neurons
        self.priority = mk_priority_map(domain, seed)

        # TODO: special case initial state generation when vertices in domain
        # are of a similar order as the number of neurons (use a reservoir method).

# notes for later:
# tip datastructure: list (indexed by nid -1) of:  list or np array of tip type
# tip type: (vertex, former edge index | stymied)
#
# to generalize to parallel initialization, use priority scheme to resolve
# collisions in initialization; maintain list of evicted or frustrated neuron ids;
# retry until all resolved (or give up if iteration count starts getting 'large').

        self._tips = [] * self._n

        retry = []
        for i in range(0, n_neurons):
            nid = i + 1
            v = 





    def occupy(self, vertex, nid, parity, edge):
        # Can provisionally add neuron nid to vertex if there is an EMPTY slot, or if
        # the least priority of a slot with same parity is less than that of (vertex, nid).
        # Return None on failure, True if occupied an empty slot, or the nid of evicted
        # neuron if the slot was occupied.

        v_hash = self.domain.vertex_hash(vertex)
        mo = self.domain.max_occupancy
        up = Domain.unpack(self.domain.get(vertex))

        min_priority = 2**64 - 1
        min_slot = -1
        for s in range(mo):
            if up[s].occupancy == Domain.Occupancy.EMPTY:
                up[s] = (nid, parity, edge)
                domain.put(vertex, s, Domain.pack(up[s]))
                return True
            elif up[s].occupancy == parity:
                p = self.priority(up[s].nid, v_hash)
                if p <= min_priority:
                    min_priority = p
                    min_slot = s

        if min_slot<0:
            return None
        else:
            p = self.priority(nid, v_hash)
            evicted = up[min_slot].nid
            if (p, nid) > (min_priority, evicted):
                up[min_slot] = (nid, parity, edge)
                domain.put(vertex, min_slot, Domain.pack(up[min_slot]))
                return evicted
            else:
                return None


