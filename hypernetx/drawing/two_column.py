# Copyright © 2018 Battelle Memorial Institute
# All rights reserved.

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle

import networkx as nx

from .util import get_frozenset_label


def layout_two_column(H, spacing=0, width=1, flip_orientation=False, nodeorder=None):
    """
    Two column (bipartite) layout algorithm.

    This algorithm first converts the hypergraph into a bipartite graph and
    then computes connected components. Disonneccted components are handled
    independently and then stacked together.

    Within a connected component, the spectral ordering of the bipartite graph
    provides a quick and dirty ordering that minimizes edge crossings in the
    diagram.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    spacing: float
        amount of whitespace between disconnected components
    width : float
        width between column of edges and column of nodes
    """
    offset = 0
    pos = {}
    def stack(vertices, x, height, flip_orientation):
        for i, v in enumerate(vertices):
            if flip_orientation:
                pos[v] = (i + offset + (height - len(vertices)) / 2, x)
            else:
                pos[v] = (x, i + offset + (height - len(vertices)) / 2)

    G = H.bipartite()
    for ci in nx.connected_components(G): # iterates on connected groups of nodes
        Gi = G.subgraph(ci)

        if nodeorder is None:
            key = {v: i for i, v in enumerate(nx.spectral_ordering(Gi))}.get
        else:
            edge_names = [edge for edge in Gi.nodes if isinstance(edge, str)] # edges are kept as str while nodes are int
            edgeorder = {edge_names[n] : n for n in range(len(edge_names))}
            key = {**nodeorder, **edgeorder}.get

        ci_vertices, ci_edges = [
            sorted([v for v, d in Gi.nodes(data=True) if d["bipartite"] == j], key=key)
            for j in [0, 1]
        ]

        height = max(len(ci_vertices), len(ci_edges))
        stack(ci_vertices, 0, height, flip_orientation=flip_orientation)
        stack(ci_edges, width, height, flip_orientation=flip_orientation)

        offset += height + spacing

    return pos


def draw_hyper_edges(H, pos, ax=None, edgecolor=None, nodecolor=None, nodesize=0.1, **kwargs):
    """
    Renders hyper edges for the two column layout.

    Each node-hyper edge membership is rendered as a line connecting the node
    in the left column to the edge in the right column.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments passed to matplotlib.LineCollection

    Returns
    -------
    LineCollection
        the hyper edges
    """
    ax = ax or plt.gca()

    pairs = [(node, edge.uid) for edge in H.edges() for node in edge] # list of (node, edge_id) pairs

    kwargs = {
        k: v if type(v) != dict else [v.get(e) for _, e in pairs]
        for k, v in kwargs.items()
    }

    lines = LineCollection([(pos[node], pos[edge_id]) for node, edge_id in pairs], color=edgecolor, zorder=0, **kwargs)
    ax.add_collection(lines)

    # add nodes as circles
    nodes = list(H.nodes)
    patches = [Circle(pos[node], nodesize) for node in nodes]

    if nodecolor is None:
        nodecolor = 'red'
    if isinstance(nodecolor, dict):
        nodecolor =[nodecolor[node] for node in nodes]
    ax.add_collection(PatchCollection(patches, color=nodecolor))

    return lines

def draw_hyper_column_labels(
    H, pos, labels={}, with_node_labels=True, with_edge_labels=True, ax=None
):
    """
    Renders hyper labels (nodes and edges) for the two column layout.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    labels: dict
        custom labels for nodes and edges can be supplied
    with_node_labels: bool
        False to disable node labels
    with_edge_labels: bool
        False to disable edge labels
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments passed to matplotlib.LineCollection

    """

    ax = ax or plt.gca()

    edges = [e.uid for e in H.edges()]

    to_draw = []
    if with_node_labels:
        to_draw.append((H.nodes(), "right"))

    if with_edge_labels:
        to_draw.append((H.edges(), "left"))

    for points, ha in to_draw:
        for p in points:
            ax.annotate(labels.get(p.uid, p.uid), pos[p.uid], ha=ha, va="center")

def draw_hyper_row_labels(
    H, pos, labels={}, with_node_labels=True, with_edge_labels=True, ax=None
):
    """
    Renders hyper labels (nodes and edges) for the two column layout.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    pos: dict
        mapping of node and edge positions to R^2
    labels: dict
        custom labels for nodes and edges can be supplied
    with_node_labels: bool
        False to disable node labels
    with_edge_labels: bool
        False to disable edge labels
    ax: Axis
        matplotlib axis on which the plot is rendered
    kwargs: dict
        keyword arguments passed to matplotlib.LineCollection

    """

    ax = ax or plt.gca()

    edges = [e.uid for e in H.edges()]

    to_draw = []
    if with_node_labels:
        to_draw.append((H.nodes(), "top"))

    if with_edge_labels:
        to_draw.append((H.edges(), "bottom"))

    for points, va in to_draw:
        for p in points:
            ax.annotate(labels.get(p.uid, p.uid), (pos[p.uid][0], pos[p.uid][1]-0.15), ha="center", va=va)

def draw(
    H,
    with_node_labels=True,
    with_edge_labels=True,
    with_node_counts=False,
    with_edge_counts=False,
    with_color=True,
    edge_kwargs=None,
    ax=None,
    column_spacing=1,
    flip_orientation=False,
    edgecolor=None,
    nodecolor=None,
    nodeorder=None,
    nodesize=0.1
):
    """
    Draw a hypergraph using a two-collumn layout.

    This is intended reproduce an illustrative technique for bipartite graphs
    and hypergraphs that is typically used in papers and textbooks.

    The left column is reserved for nodes and the right column is reserved for
    edges. A line is drawn between a node an an edge

    The order of nodes and edges is optimized to reduce line crossings between
    the two columns. Spacing between disconnected components is adjusted to make
    the diagram easier to read, by reducing the angle of the lines.

    Parameters
    ----------
    H: Hypergraph
        the entity to be drawn
    with_node_labels: bool
        False to disable node labels
    with_edge_labels: bool
        False to disable edge labels
    with_node_counts: bool
        set to True to label collapsed nodes with number of elements
    with_edge_counts: bool
        set to True to label collapsed edges with number of elements
    with_color: bool
        set to False to disable color cycling of hyper edges
    edge_kwargs: dict
        keyword arguments to pass to matplotlib.LineCollection
    ax: Axis
        matplotlib axis on which the plot is rendered
    """
    edge_kwargs = edge_kwargs or {}

    ax = ax or plt.gca()

    pos = layout_two_column(H, width=column_spacing, flip_orientation=flip_orientation, nodeorder=nodeorder)

    V = [v.uid for v in H.nodes()]
    E = [e.uid for e in H.edges()]

    labels = {}
    labels.update(get_frozenset_label(V, count=with_node_counts))
    labels.update(get_frozenset_label(E, count=with_edge_counts))

    if with_color:
        edge_kwargs["color"] = {
            e.uid: plt.cm.tab10(i % 10) for i, e in enumerate(H.edges())
        }

    draw_hyper_edges(H, pos, ax=ax, edgecolor=edgecolor, nodecolor=nodecolor, nodesize=nodesize, **edge_kwargs)

    if flip_orientation:
        draw_hyper_row_labels(H, pos, labels, ax=ax, with_node_labels=with_node_labels, with_edge_labels=with_edge_labels)
    else:
        draw_hyper_column_labels(H, pos, labels, ax=ax, with_node_labels=with_node_labels, with_edge_labels=with_edge_labels)

    ax.set_aspect(1)
    ax.autoscale_view(scaley=True, scalex=True)

    # print(ax.get_xlim())
    # print(ax.get_ylim())

    ax.axis("off")
