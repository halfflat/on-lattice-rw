#!/usr/bin/env python

import argparse as ap
import numpy as np
import periodic_net as P
import plotly.io as pio
import plotly.graph_objs as pgo
import sys


# vertices and edges by coordinate

def plot_net(net, extent):
    def lex_lt(a, b):
        i = np.argmin(a==b)
        return a[i]<b[i]

    extent += (net.n_vertices,)
    N = np.prod(np.asarray(extent))
    vs = np.empty((N, net.dimension))
    vr = 0

    es = np.empty((3*int(N*net.degree/2), net.dimension))
    er = 0
    esep = np.full((net.dimension,), np.nan)

    for i in np.ndindex(extent):
        v = np.asarray(i)

        vpos = net.position(v)
        vs[vr] = vpos
        vr += 1

        for e in net.edge_indices_from(v[-1]):
            u  = net.traverse(v, e)
            if lex_lt(u, v) or np.any(u >= extent) or np.any(u < 0): continue

            upos = net.position(u)
            es[er] = vpos
            es[er+1] = upos
            es[er+2,:] = esep
            er += 3

    if net.dimension == 2:
        trace_v = pgo.Scatter(x=vs[:,0], y=vs[:,1], mode='markers', marker=dict(symbol='circle'), hoverinfo='skip')
        trace_e = pgo.Scatter(x=es[:er,0], y=es[:er,1], mode='lines', line=dict(width=1), hoverinfo='skip')
        return pgo.Figure(data=[trace_v, trace_e])
    elif net.dimension == 3:
        trace_v = pgo.Scatter3d(x=vs[:,0], y=vs[:,1], z=vs[:,2], mode='markers', marker=dict(symbol='circle'), hoverinfo='skip')
        trace_e = pgo.Scatter3d(x=es[:er,0], y=es[:er,1], z=es[:er,2], mode='lines', line=dict(width=1), hoverinfo='skip')
        return pgo.Figure(data=[trace_v, trace_e])
    else:
        return None


def main():
    nets = [k for k, v in vars(P).items() if isinstance(v, P.PeriodicNet)]

    def intsplit(s):
        return tuple([int(f) for f in s.split(',')])

    parser = ap.ArgumentParser(prog='plot_net', description='Plot subregion of a predefined net')
    parser.add_argument('--list', action='store_true', help='list predefined nets')
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

    fig = plot_net(net, extent)
    fig.layout.template = 'simple_white'
    fig.layout.yaxis.scaleanchor = 'x'
    fig.layout.showlegend = False

    pio.renderers.default = 'browser'
    fig.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())

