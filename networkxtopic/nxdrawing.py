import matplotlib.pyplot as plt
import networkx as nx


def draw_ex():
    G = nx.dodecahedral_graph()
    nx.draw(G, pos=nx.spring_layout(G), nodecolor='r', edge_color='b', with_labels=True, data='weight')
    plt.savefig("draw_graph.png")
    plt.show()


def draw_networkx_ex():
    G = nx.dodecahedral_graph()
    nx.draw(G)
    plt.show()
    nx.draw_networkx(G, pos=nx.spring_layout(G))
    limits = plt.axis('off')
    plt.show()
    nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))
    plt.show()
    edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    plt.show()
    labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))
    plt.show()
    edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))
    plt.show()
    print("Circular layout")
    nx.draw_circular(G)
    plt.show()
    print("Random layout")
    nx.draw_random(G)
    plt.show()
    print("Spectral layout")
    nx.draw_spectral(G)
    plt.show()
    print("Spring layout")
    nx.draw_spring(G)
    plt.show()
    print("Shell layout")
    nx.draw_shell(G)
    plt.show()
    print("Graphviz")
    # nx.draw_graphviz(G)
    # plt.show()


# draw_ex()
draw_networkx_ex()
