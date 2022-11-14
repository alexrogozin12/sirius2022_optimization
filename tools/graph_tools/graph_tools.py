import random
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph, complete_graph
from networkx.generators import circulant_graph

def vanish_random_edges_from_graph(G : nx.Graph, occupancy : float = 0.0, vanish_rate : int = 0) -> np.matrix:
    if occupancy*vanish_rate != 0: raise ValueError()
    edges = list(G.edges)
    num_of_edges = len(edges)
    if occupancy != 0: vanish_rate = int(np.round(num_of_edges*occupancy))
    if num_of_edges - vanish_rate < G.number_of_nodes() - 1:
        print("vanish_rate more than N", vanish_rate, " ", G.number_of_nodes() - 1)
        return
    for i in range(vanish_rate):
        # TODO: make this more optimal
        # MB - DONE but not sure (self-loop nodes could be removed earlier, but I this that is it slower way)
        while True:
            edge = edges[random.randint(0, num_of_edges-1)]
            G.remove_edge(*edge)
            edges.remove(edge)
            num_of_edges -= 1
            if nx.is_connected(G) and edge[0] != edge[1]:
                break
            else:
                G.add_edge(edge[0], edge[1], weight=1)

    return nx.adjacency_matrix(G).todense()

def vanish_random_edges_from_matrix(M : np.matrix, occupancy : float = 0.0, vanish_rate : int = 0) -> np.matrix:
    G = nx.from_numpy_matrix(M)
    M = vanish_random_edges_from_graph(G, occupancy, vanish_rate)
    return fill_metropolis_weigts(M)

def vanish_random_edge_from_matrix(M : np.matrix) -> np.matrix:
    G = nx.from_numpy_matrix(M)
    M = vanish_random_edges_from_graph(G, occupancy=0.0, vanish_rate=1)
    return fill_metropolis_weigts(M)

def make_circular_graph_matrix(N : int) -> np.matrix:
    G = nx.circulant_graph(N, [1])
    return nx.adjacency_matrix(G).todense()

def make_complete_graph_matrix(N : int) -> np.matrix:
    G = nx.complete_graph(N)
    return nx.adjacency_matrix(G).todense()

def make_random_graph_matrix(N : int, occupancy : float = 0.0, vanish_rate : int = 0) -> np.matrix:
    G = nx.complete_graph(N)
    M = vanish_random_edges_from_graph(G, occupancy, vanish_rate)
    return fill_metropolis_weigts(M)

def fill_metropolis_weigts(M : np.matrix) -> np.matrix:
    n_links = []
    for i in range(len(M)):
        n_links.append(np.count_nonzero(M[i]))
    G = nx.from_numpy_matrix(M)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = 1 / (1 + max(n_links[u], n_links[v]))
    M = nx.adjacency_matrix(G).todense()
    for i in range(len(M)):
        G.add_edge(i, i, weight=(1 - np.sum(M[i])))
    return nx.adjacency_matrix(G).todense()

def make_graph_img(M : np.matrix, num_of_lines : int = 5, fig_size : tuple = (5, 5) ):
    G = nx.from_numpy_matrix(M)
    _w = []
    _step = 1 / num_of_lines

    for i in range(num_of_lines):
        _w.append([(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > i * _step and d["weight"] <= (i+1)*_step])

#    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility
    pos = nx.spring_layout(G)
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
