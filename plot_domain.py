import numpy as np
import math
from rw_domain import Domain
import plotly.colors as pcol
import plotly.io as pio
import plotly.graph_objs as pgo
import sys

def c0(vs): return [v[0] for v in vs]
def c1(vs): return [v[1] for v in vs]
def c2(vs): return [v[2] for v in vs]
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

def plot_domain(domain, colorscale='rainbow'):
    """Produce a plotly Figure representing a random walk domain"""

    def lex_lt(a, b):
        i = np.argmin(a==b)
        return a[i]<b[i]

    def displace(pos, slot, twist = 0.0, r = 0.05):
        if domain.max_occupancy > 1:
            a = 2*math.pi*(twist+slot/domain.max_occupancy)
            delta = (r*math.cos(a), r*math.sin(a))
            return [pos[0]+delta[0], pos[1]+delta[1]] + list(pos[2:])
        else:
            return pos

    def twist(vertex):
        f = domain.vertex_hash(vertex)/2**domain.vertex_hash_bits
        return math.modf(97*f)[0]

    dim = domain.net.dimension
    extent = domain.extent
    N = np.prod(np.asarray(extent))

    cmap = vdC_colorscale(colorscale)

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
        vnvs = { n: displace(vpos, s, twist=twist(v)) for (s, n) in enumerate(vocc.nid) if n>0 }
        for n, pos in vnvs.items():
            if n not in nvs: nvs[n] = []
            nvs[n].append(pos)

        for u, _ in domain.neighbours(v):
            if domain.is_border(u): continue

            upos = domain.position(u)
            if not lex_lt(u, v):
                es.extend([vpos, upos, esep])

            uocc = Domain.unpack(domain.get(u))
            for s, n in enumerate(uocc.nid):
                if n in vnvs and tuple(domain.traverse_r(u, uocc.prev[s]))==tuple(v):
                    if n not in nes: nes[n] = []
                    nes[n].extend([vnvs[n], displace(upos, s, twist=twist(u)), esep])

    if dim == 2:
        traces = []
        # net vertices and edges
        traces.append(pgo.Scatter(x=c0(vs), y=c1(vs), mode='markers', marker=dict(symbol='circle', size=2), hoverinfo='skip'))
        traces.append(pgo.Scatter(x=c0(es), y=c1(es), mode='lines', line=dict(width=1, dash='dot'), hoverinfo='skip'))
        # neuron vertices and edges
        for n, vv in nvs.items():
            traces.append(pgo.Scatter(x=c0(vv), y=c1(vv), mode='markers', marker=dict(symbol='circle', size=6, color=cmap(n)), hoverinfo='skip'))
        for n, ee in nes.items():
            traces.append(pgo.Scatter(x=c0(ee), y=c1(ee), mode='lines', line=dict(width=1, color=cmap(n)), hoverinfo='skip'))
        return pgo.Figure(data=traces)
    elif dim == 3:
        traces = []
        # net vertices and edges
        traces.append(pgo.Scatter3d(x=c0(vs), y=c1(vs), z=c2(vs), mode='markers', marker=dict(symbol='circle', size=2), hoverinfo='skip'))
        #traces.append(pgo.Scatter3d(x=c0(es), y=c1(es), z=c2(es), mode='lines', line=dict(width=1, dash='dot'), hoverinfo='skip'))
        traces.append(pgo.Scatter3d(x=c0(es), y=c1(es), z=c2(es), mode='lines', line=dict(width=1, color='#e0e0e0'), hoverinfo='skip'))
        # neuron vertices and edges
        for n, vv in nvs.items():
            traces.append(pgo.Scatter3d(x=c0(vv), y=c1(vv), z=c2(vv), mode='markers', marker=dict(symbol='circle', size=6, color=cmap(n)), hoverinfo='skip'))
        for n, ee in nes.items():
            traces.append(pgo.Scatter3d(x=c0(ee), y=c1(ee), z=c2(ee), mode='lines', line=dict(width=4, color=cmap(n)), hoverinfo='skip'))
        return pgo.Figure(data=traces)
    else:
        return None

def render_figure(fig):
    fig.layout.template = 'simple_white'
    fig.layout.yaxis.scaleanchor = 'x'
    fig.layout.showlegend = False

    pio.renderers.default = 'browser'
    fig.show()

