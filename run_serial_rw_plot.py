#!/usr/bin/env python

from rw_domain import Domain
from rw_serial import State
from plot_domain import plot_domain, render_figure
import periodic_net as P
import numpy as np
import argparse as ap
import sys

def main():
    nets = [k for k, v in vars(P).items() if isinstance(v, P.PeriodicNet)]

    def intsplit(s):
        return tuple([int(f) for f in s.split(',')])

    parser = ap.ArgumentParser(prog='plot_net', description='Plot subregion of a predefined net')
    parser.add_argument('--list', action='store_true', help='list predefined nets')
    parser.add_argument('--mo', help='maximum occupancy of vertices', type=int, default=1)
    parser.add_argument('--steps', help='maximum simulation steps', type=int, default=10000)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('-n', '--neurons', help='number of neurons', type=int, default=1)
    parser.add_argument('net', choices=nets, nargs='?', default='sql')
    parser.add_argument('extent', help='extent I,J or I,J,K of subregion', type=intsplit, nargs='?', default=(2,))

    p = parser.parse_args()
    if p.list:
        for name in nets:
            n = getattr(P, name)
            print('{}\t{}-periodic {}-net'.format(name, n.dimension, n.degree))

        return 0

    net = getattr(P, p.net)
    extent = np.pad(np.asarray(p.extent), (0,net.dimension), mode='edge')
    extent = tuple(extent[:net.dimension])


    state = State(p.neurons, net, extent, p.mo, seed=p.seed)
    for _ in range(p.steps):
        if not state.step(): break

    render_figure(plot_domain(state.domain))

if __name__ == '__main__':
    sys.exit(main())
