import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph, complete_graph
from networkx.generators import circulant_graph

def make_cirvular_graph(N):
    G = nx.circulant_graph(N, [1])
    return nx.adjacency_matrix(G).todense()

def make_complete_graph(N):
    G = nx.complete_graph(N)
    return nx.adjacency_matrix(G).todense()

def make_random_graph(N, occupancy=0.5):
    G = nx.complete_graph(N)
    fixed_edges = []
    edges = list(G.edges)
    num_of_edges = len(edges)
    vanish_rate = int(np.round(num_of_edges*occupancy))
    if vanish_rate < N:
        print("vanish_rate lover than N", vanish_rate, " ", N)
        return
    for i in range(vanish_rate):
        # TODO: make this more optimal
        while True:
            edge = edges[random.randint(0, num_of_edges-1)]
            if edge not in fixed_edges:
                G.remove_edge(edge[0], edge[1])
            if nx.is_connected(G):
                edges.remove(edge)
                num_of_edges -= 1
                break
            else:
                G.add_edge(edge, weight=1)
                fixed_edges.append(edge)
    return nx.adjacency_matrix(G).todense()

def vanish_random_edge(M):
    G = nx.from_numpy_matrix(M)
    fixed_edges = []
    edges = list(G.edges)
    num_of_edges = len(edges)
    # TODO: make this more optimal
    while True:
        edge = edges[random.randint(0, num_of_edges-1)]
        if edge not in fixed_edges:
            G.remove_edge(edge[0], edge[1])
        if nx.is_connected(G):
            edges.remove(edge)
            num_of_edges -= 1
            break
        else:
            G.add_edge(edge[0], edge[1], weight=1)
            fixed_edges.append(edge)
    return fill_metropolis_weigts(nx.adjacency_matrix(G).todense())

def fill_metropolis_weigts(M):
    n_links = []
    for i in range(len(M)):
        n_links.append(np.count_nonzero(M[i]))
    G = nx.from_numpy_matrix(M)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = 1 / max(n_links[u], n_links[v])
    return nx.adjacency_matrix(G).todense()

def make_graph_img(M, num_of_lines = 5, fig_size=(5, 5)):
    G = nx.from_numpy_matrix(M)
    _w = []
    _step = 1 / num_of_lines
    w_08 = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.8]
    for i in range(num_of_lines):
        _w.append([(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > i * _step and d["weight"] <= (i+1)*_step])
    
    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    edge_labels = nx.get_edge_attributes(G, "weight")
    for key in edge_labels:
        edge_labels[key] = np.round(edge_labels[key], 2)

    plt.figure(figsize=fig_size, dpi=80)

    nx.draw_networkx_nodes(G, pos, node_size=700)

    for i in range(num_of_lines):
        nx.draw_networkx_edges(G, pos, edgelist=_w[i], width=4, alpha=(i+1)*_step)

    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
