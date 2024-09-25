import numpy as np
import math
from rw_domain import Domain
import plotly.colors as pcol
import plotly.io as pio
import plotly.graph_objs as pgo
import sys


def c0(vs): return [v[0] for v in vs]
def c1(vs): return [v[1] for v in vs]
def ce(vs): return [v[-1] for v in vs]

class vdC_colorscale:
    def __init__(self, colorscale):
        if type(colorscale)is str:
            self.colorscale = pcol.get_colorscale(colorscale)
        else:
            self.colorscale = colorscale
        self._colors = []

    def vdC(n):
        if n==0: return 0
        p = 0
        r = 1
        while n>0:
            p = (p<<1) + (n&1)
            r += 1
            n = n>>1
        return p/(1<<(r-1))

    def __call__(self, k):
        if k>=len(self._colors):
            self._colors.extend(pcol.sample_colorscale(self.colorscale, [vdC_colorscale.vdC(j) for j in range(len(self._colors), k+1)]))
        return self._colors[k]

# vertices and edges by coordinate

def plot_domain(domain):
    def lex_lt(a, b):
        i = np.argmin(a==b)
        return a[i]<b[i]

    if domain.max_occupancy>1:
        r = 0.05
        a = 2*math.pi/domain.max_occupancy
        displacements = [(r*math.cos(i*a), r*math.sin(i*a)) for i in range(0, domain.max_occupancy)]

    def displace(pos, i):
        d = displacements[i]
        return [pos[0]+d[0], pos[1]+d[1]] + list(pos[2:])

    dim = domain.net.dimension
    extent = domain.extent
    N = np.prod(np.asarray(extent))

    cmap = vdC_colorscale('picnic')

    # vertices and edges of underlying net
    vs = []
    es = []
    esep = np.full((dim,), np.nan)

    # vertices and edges of neurons (dictionaries nid -> vertices, nid -> edges)
    nvs = {}
    nes = {}

    for i in np.ndindex(tuple(extent)):
        v = np.asarray(i)

        vpos = domain.position(v)
        vs.append(vpos)

        vocc = Domain.unpack(domain.get(v))
        vnvs = { n: displace(vpos, s) for (s, n) in enumerate(vocc.nid) if n>0 }
        for n, pos in vnvs.items():
            if n not in nvs: nvs[n] = []
            nvs[n].append(pos)

        for u, _ in domain.neighbours(v):
            if lex_lt(u, v) or domain.is_border(u): continue

            upos = domain.position(u)
            es.extend([vpos, upos, esep])

            uocc = Domain.unpack(domain.get(u))
            unvs = { n: displace(upos, s) for (s, n) in enumerate(uocc.nid) if n>0 }
            for n, pos in unvs.items():
                if n in vnvs:
                    if n not in nes: nes[n] = []
                    nes[n].extend([vnvs[n], unvs[n], esep])

    if dim == 2:
        traces = []
        # net vertices and edges
        traces.append(pgo.Scatter(x=c0(vs), y=c1(vs), mode='markers', marker=dict(symbol='circle'), hoverinfo='skip'))
        traces.append(pgo.Scatter(x=c0(es), y=c1(es), mode='lines', line=dict(width=1), hoverinfo='skip'))
        # neuron vertices and edges
        for n, vv in nvs.items():
            traces.append(pgo.Scatter(x=c0(vv), y=c1(vv), mode='markers', marker=dict(symbol='circle', color=cmap(n)), hoverinfo='skip'))
        for n, ee in nes.items():
            traces.append(pgo.Scatter(x=c0(ee), y=c1(ee), mode='lines', line=dict(width=1, color=cmap(n)), hoverinfo='skip'))

        return pgo.Figure(data=traces)
    elif dim == 3:
        return None
    else:
        return None

def run_domain(domain):
    fig = plot_domain(domain)
    fig.layout.template = 'simple_white'
    fig.layout.yaxis.scaleanchor = 'x'
    fig.layout.showlegend = False

    pio.renderers.default = 'browser'
    fig.show()

#def main():
#    nets = [k for k, v in vars(P).items() if isinstance(v, P.PeriodicNet)]
#
#    def intsplit(s):
#        return tuple([int(f) for f in s.split(',')])
#
#    parser = ap.ArgumentParser(prog='plot_net', description='Plot subregion of a predefined net')
#    parser.add_argument('--list', action='store_true', help='list predefined nets')
#    parser.add_argument('net', choices=nets, nargs='?', default='sql')
#    parser.add_argument('extent', help='extent I,J or I,J,K of subregion', type=intsplit, nargs='?', default=(2,))
#
#    p = parser.parse_args()
#    if p.list:
#        for name in nets:
#            n = getattr(P, name)
#            print('{}\t{}-periodic {}-net'.format(name, n.dimension, n.degree))
#
#        return 0
#
#    net = getattr(P, p.net)
#
#    extent = np.pad(np.asarray(p.extent), (0,net.dimension), mode='edge')
#    extent = tuple(extent[:net.dimension])
#
#    fig = plot_net(net, extent)
#    fig.layout.template = 'simple_white'
#    fig.layout.yaxis.scaleanchor = 'x'
#    fig.layout.showlegend = False
#
#    pio.renderers.default = 'browser'
#    fig.show()
#    return 0
#
#if __name__ == '__main__':
#    sys.exit(main())
#
