'''Networkx tutorial - algorithms'''

import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.isomorphism as iso
from networkx.algorithms import approximation as approx, bipartite
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_st_node_cut
from networkx.algorithms.tree import branching_weight, maximum_branching, minimum_branching, \
    maximum_spanning_arborescence, minimum_spanning_arborescence
from networkx.algorithms.triads import triadic_census


class OrderedNodeGraph(nx.Graph):
    node_dict_factory = OrderedDict

G1 = nx.Graph()
G2 = nx.DiGraph()
G3 = nx.MultiGraph()
G4 = nx.MultiDiGraph()


def draw_graph(g):
    nx.draw(g, nodecolor='r', edge_color='b', with_labels=True, data='weight')
    # nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color='b', with_labels=True )
    plt.savefig("repr_graph_1.png")
    plt.show()


def print_iter_graph_with_attribute(G, g_attribute):
    for n, nbrsdict in G.adjacency_iter():
        for nbr, eattr in nbrsdict.items():
            if g_attribute in eattr:
                print(n, nbr, eattr[g_attribute])


def print_iter_graph(g):
    for n, nbrsdict in g.adjacency_iter():
        for nbr in nbrsdict.items():
            print(n, nbr)


def graph_create():
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3, weight=0.9)
    g.add_edge('y', 'x', function=math.cos)
    elist = [('a', 'b', 5.0), ('b', 'c', 3.0), ('a', 'c', 1.0), ('c', 'd', 7.3)]
    g.add_weighted_edges_from(elist)


def dijkstra_ex1():
    G = nx.Graph()
    e = [('a', 'b', 0.3), ('b', 'c', 0.9), ('a', 'c', 0.5), ('c', 'd', 1.2)]
    G.add_weighted_edges_from(e)
    G1 = nx.path_graph(4)
    print("Shortest path:" + repr(nx.dijkstra_path(G, 'a', 'd')))
    print("Edges: " + repr(G.edges()))
    print("Nodes: " + repr(G.nodes()))
    # draw the graph
    # G=nx.cubical_graph()
    # G.add_weighted_edges_from(e)
    # nx.draw(G1)
    nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color='b', with_labels=True)
    plt.savefig("path_graph2.png")
    plt.show()


def repr_graph():
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    print("Graph::" + repr(G.edges()))
    print("Adj graph::" + repr(G.adj))


def repr_graph_1():
    G = nx.Graph()
    G.add_edge(1, 2, color='red', weight=0.84, size=300)
    print("Nodes::" + repr(G.nodes()))
    print("Size::" + repr(G[1][2]['size']))
    # nx.draw(G, pos=nx.spectral_layout(G), nodecolor='r', edge_color='b', with_labels=True)
    nx.draw(G)
    plt.savefig("repr_graph_1.png")
    plt.show()


def repr_graph_2():
    G = nx.Graph()
    G.add_node(1)
    print("Nodes::" + repr(G.nodes()))
    G.add_nodes_from([2, 3])
    G.add_nodes_from(range(100, 110))
    H = nx.Graph()
    H.add_path([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    G.add_nodes_from(H)
    draw_graph(G)


def repr_graph_3():
    G = nx.Graph()
    G.add_node(1, time='5pm')
    G.add_nodes_from([3], time='1pm')
    G.add_nodes_from([2, 4, 5])
    G.node[1]['room'] = 512
    G.add_edge(1, 2, weight=4.7)
    G.add_edges_from([(3, 4), (4, 5)])
    G.add_edges_from([(2, 3, {'weight': 8})])
    print("Nodes::" + repr(G.nodes(data=True)))
    print("Graph::" + repr(G.graph))
    print("Graph node = " + '{}'.format(1) + repr(G.node[1]))
    print("Graph node = " + '{}'.format(3) + repr(G.node[3]))
    print("Edges::" + repr(G.edges(data='weight')))
    print_iter_graph_with_attribute(G, 'weight')
    draw_graph(G)


def create_ordered_node_graph():
    ''' Create a graph object that tracks the order nodes are added'''
    G = OrderedNodeGraph()
    G.add_nodes_from((2, 1))
    print("First nodes added:" + repr(G.nodes()))
    G.add_edges_from(((2, 2), (2, 1), (1, 1)))
    print("Edges added:" + repr(G.edges()))


def connectivity_alg():
    ''' approximation of node connectivity algorithms'''
    G = nx.Graph()
    G.add_nodes_from(('a', 'b', 'c', 'd'))
    elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd')]
    G.add_edges_from(elist)
    GP = nx.petersen_graph()
    print('Print the graph')
    print_iter_graph(G)
    print('All graph connectivity')
    print(nx.all_pairs_node_connectivity(G))
    print('Connectivity between two nodes')
    print(approx.local_node_connectivity(G, 'a', 'c'))
    print('Nodes Connectivity')
    print(approx.node_connectivity(G))
    draw_graph(G)
    k_components = approx.k_components(G)
    print('k-components {}'.format(k_components))
    # draw_graph(G)


def clique_alg():
    G = nx.Graph()
    G.add_nodes_from(('a', 'b', 'c', 'd', 'e', 'f', 'g'))
    G.remove_node('d')
    elist = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('e', 'f'), ('e', 'g'), ('f', 'g'), ('c', 'g')]
    G.add_edges_from(elist)
    # G.remove_edge('c', 'd')
    print('Print the graph')
    print_iter_graph(G)
    # draw_graph(G)
    print('Cliques {}'.format(approx.max_clique(G)))


def clique_alg1():
    g = nx.karate_club_graph()
    draw_graph(g)
    '''Find all cliques of 4 or more nodes:'''
    cliques = nx.find_cliques(g)
    print('Cliques {}'.format(cliques))
    cliques4 = [clq for clq in cliques if len(clq) >= 5]
    print('Cliques of {} or more{}'.format(5, cliques4))
    '''Create a subgraph of g from all sufficiently large cliques:'''
    nodes = [n for clq in cliques4 for n in clq]
    h = g.subgraph(nodes)
    '''Through away nodes of h which have degree less than 4:'''
    deg = nx.degree(h)
    nodes = [n for n in nodes if deg[n] >= 5]
    '''The desired graph k is the subgraph of h with these nodes:'''
    k = h.subgraph(nodes)
    draw_graph(k)


def average_clustering_alg():
    g = nx.karate_club_graph()
    draw_graph(g)
    print('Average clustering {}'.format(approx.average_clustering(g, 200)))


def dominating_set_alg():
    '''A `dominating set`_[1] for an undirected graph *G with vertex set V and edge set E is a subset D of V such that
    every vertex not in D is adjacent to at least one member of D.
    An `edge dominating set`_[2] is a subset *F of E such that every edge not in F is incident to an endpoint of at
    least one edge in F.'''
    g = nx.karate_club_graph()
    draw_graph(g)
    print('Node dominated set {}'.format(approx.min_weighted_dominating_set(g)))
    print('Edge dominated set {}'.format(approx.min_edge_dominating_set(g)))


def independent_set_alg():
    '''Independent set or stable set is a set of vertices in a graph, no two of which are adjacent.
    That is, it is a set I of vertices such that for every two vertices in I, there is no edge connecting the two. '''
    g = nx.karate_club_graph()
    draw_graph(g)
    print('Max independent set {}'.format(approx.maximum_independent_set(g)))


def matching_alg():
    '''Given a graph G = (V,E), a matching M in G is a set of pairwise non-adjacent edges;
    that is, no two edges share a common vertex.'''
    g = nx.karate_club_graph()
    draw_graph(g)
    G = nx.Graph()
    G.add_nodes_from(('a', 'b', 'c', 'd', 'e', 'f', 'g'))
    elist = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('e', 'f'), ('e', 'g'), ('f', 'g'), ('c', 'g')]
    G.add_edges_from(elist)
    draw_graph(G)
    print('Min maximum matching set {}'.format(approx.min_maximal_matching(G)))


def ramsey_alg():
    '''Calculates ramsey number, returns clique and independent set '''
    g = nx.karate_club_graph()
    draw_graph(g)
    print('Ramsey R2 {}'.format(approx.ramsey_R2(g)))


def vertex_cover_alg():
    '''Given an undirected graph G = (V, E) and a function w assigning nonnegative weights to its vertices, find a
    minimum weight subset of V such that each edge in E is incident to at least one vertex in the subset.'''
    g = nx.karate_club_graph()
    draw_graph(g)
    print('Min vertex cover {}'.format(approx.min_weighted_vertex_cover(g)))


def assortativity_alg():
    '''Assortativity measures the similarity of connections in the graph with respect to the node degree.'''
    g = nx.karate_club_graph()
    draw_graph(g)
    print('Degree asssortativity coefficient {}'.format(nx.degree_assortativity_coefficient(g)))
    print('Average neighbour degree {}'.format(nx.average_neighbor_degree(g)))
    print('Average degree connectivity {}'.format(nx.average_degree_connectivity(g)))


def bipartite_alg():
    ''''''
    B = nx.Graph()
    B.add_nodes_from([1, 2, 3, 4], bipartite=0)  # Add the node attribute "bipartite"
    B.add_nodes_from(['a', 'b', 'c'], bipartite=1)
    B.add_edges_from([(1, 'a'), (1, 'b'), (2, 'b'), (2, 'c'), (3, 'c'), (4, 'a')])
    draw_graph(B)
    nx.draw
    if (nx.is_connected(B)):
        bottom_nodes, top_nodes = bipartite.sets(B)
        print('Botton modes {}'.format(bottom_nodes))
        print('Top nodes {}'.format(top_nodes))
        '''other way of selecting the top and bottom nodes from B'''
        top_nodes = set(n for n, d in B.nodes(data=True) if d['bipartite'] == 0)
        bottom_nodes = set(B) - top_nodes


def bipartite_alg1():
    ''''''
    RB = bipartite.random_graph(5, 7, 0.2)
    draw_graph(RB)
    RB_top = set(n for n, d in RB.nodes(data=True) if d['bipartite'] == 0)
    RB_bottom = set(RB) - RB_top
    print('RB_top {}'.format(list(RB_top)))
    print('RB_botton {}'.format(list(RB_bottom)))
    print('RB Density {}'.format(bipartite.density(RB, (RB_top, RB_bottom))))


def bipartite_alg2():
    ''''''
    G = nx.complete_bipartite_graph(3, 2)
    draw_graph(G)
    X = set([0, 1, 2])
    print('Density X {}'.format(bipartite.density(G, X)))
    Y = set([3, 4])
    print('Density Y {}'.format(bipartite.density(G, Y)))
    degX, degY = bipartite.degrees(G, Y)
    print('Density X {}'.format(degX))
    print('Density Y {}'.format(degY))
    print('Maximum matching {}'.format(bipartite.maximum_matching(G)))
    # print('Biadjacecy matrix {}'.format(bipartite.biadjacency_matrix(G)))


def bipartite_projections():
    ''''''
    B = nx.path_graph(4)
    g = nx.karate_club_graph()
    draw_graph(g)
    G = bipartite.projected_graph(g, [1, 3])
    draw_graph(G)
    print(G.nodes())
    print(G.edges())


def bipartite_spectral_bipartivity():
    '''calculates the bipartite coefficient'''
    G = nx.path_graph(4)
    K = nx.karate_club_graph()
    print('Spectral bipartivity coefficient {}'.format(bipartite.spectral_bipartivity(K)))


def bipartite_clustering():
    '''The bipartie clustering coefficient is a measure of local density of connections'''
    G = nx.path_graph(14)
    print('Clustering coefficient {}'.format(bipartite.clustering(G, nodes=(1, 3), mode='max')))
    print('Average clustering coefficient {}'.format(bipartite.clustering(G, mode='max')))
    print('Latapy clustering coefficient  {}'.format(bipartite.latapy_clustering(G, nodes=(1, 3), mode='max')))
    print('Robin Alexander clustering coefficient {}'.format(bipartite.robins_alexander_clustering(G)))


def bipartite_node_redundancy():
    '''The redundancy coefficient of a node v is the fraction of pairs of neighbors of v that are both linked
     to other nodes.'''
    G = nx.cycle_graph(4)
    G1 = nx.path_graph(15)
    G2 = nx.karate_club_graph()
    draw_graph(G2)
    print('Node redundancy coefficient {}'.format(bipartite.node_redundancy(G2, nodes=(24, 33))))
    '''Compute the average redundancy for the graph:'''
    # print('Average redundancy for graph'.format(sum(bipartite.node_redundancy(G2).values()) / len(G2)))
    '''Compute the average redundancy for a set of nodes:'''
    rc = bipartite.node_redundancy(G2, nodes=[24, 33])
    nodes = [24, 33]
    print('Average redundancy for graph {}'.format(sum(rc[n] for n in nodes) / len(nodes)))


def bipartite_centrality():
    '''Compute the closeness centrality for nodes in a bipartite network.'''
    '''The closeness of a node is the distance to all other nodes in the graph or in the case that the graph
    is not connected to all other nodes in the connected component containing that node.'''
    G = nx.complete_bipartite_graph(3, 2)
    G1 = nx.path_graph(15)
    draw_graph(G1)
    print('Closeness centrality {}'.format(bipartite.closeness_centrality(G1, nodes=[0])))
    '''The degree centrality for a node v is the fraction of nodes connected to it.'''
    print('Degree centrality {}'.format(bipartite.degree_centrality(G1, nodes=[0])))
    '''Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v.'''
    print('Betweenness centrality {}'.format(bipartite.betweenness_centrality(G1, nodes=[0, 1, 2, 3])))


def bipartite_graph_generators():
    '''graph generator functions'''
    G1 = nx.complete_bipartite_graph(5, 8)
    # draw_graph(G1)
    # z = [int(random.gammavariate(alpha=5.0, beta=1.0)) for i in range(100)]
    # G = nx.configuration_model(z)
    # draw_graph(G)
    HH = bipartite.havel_hakimi_graph(aseq=[5, 3, 2, 1, 1], bseq=[3, 3, 3, 2, 1])
    # draw_graph(HH)
    PAG = bipartite.preferential_attachment_graph(aseq=[2, 2, 1, 1], p=0.7)
    # draw_graph(PAG)
    GNMK = bipartite.gnmk_random_graph(n=5, m=5, k=15)
    draw_graph(GNMK)


def blockmodel_alg():
    '''The blockmodel technique collapses nodes into blocks based on a given partitioning of the node set.
    Each partition of nodes (block) is represented as a single node in the reduced graph.'''
    g = nx.path_graph(6)
    partition = [[0, 1], [2, 3], [4, 5]]
    draw_graph(g)
    m = nx.blockmodel(g, partition)
    draw_graph(m)
    k = nx.karate_club_graph()
    draw_graph(k)
    # draw_graph(nx.blockmodel(K, [[0,12]]))


def boundary_alg():
    '''Edge boundaries are edges that have only one end in the set of nodes.
    Node boundaries are nodes outside the set of nodes that have an edge to a node in the set.'''
    k = nx.karate_club_graph()
    draw_graph(k)
    print('Edge boundary {}'.format(nx.edge_boundary(k, nbunch1=[30, 31, 32])))
    print('Node boundary {}'.format(nx.node_boundary(k, nbunch1=[30, 31, 32])))


def centrality_alg():
    """Degree"""
    K = nx.karate_club_graph()
    draw_graph(K)
    G = nx.DiGraph()
    G.add_nodes_from(('a', 'b', 'c', 'd', 'e', 'f', 'g'))
    elist = [('a', 'b'), ('a', 'c'), ('b', 'c'), ('e', 'f'), ('e', 'g'), ('f', 'g'), ('c', 'g')]
    G.add_edges_from(elist)
    draw_graph(G)
    """ The degree centrality for a node v is the fraction of nodes it is connected to."""
    print('Degree centrality {}'.format(nx.degree_centrality(G)))
    '''The in-degree centrality for a node v is the fraction of nodes its incoming edges are connected to.'''
    print('In-degree centrality {}'.format(nx.in_degree_centrality(G)))
    '''The out-degree centrality for a node v is the fraction of nodes its outgoing edges are connected to.'''
    print('Out-degree centrality {}'.format(nx.out_degree_centrality(G)))
    '''Closeness centrality'''
    '''Closeness centrality of a node u is the reciprocal of the sum of the shortest path distances from u to
    all n-1 other nodes. '''
    print('Closeness centrality {}'.format(nx.closeness_centrality(G)))
    '''Betweenness centrality of a node v is the sum of the fraction of all-pairs shortest paths that pass through v'''
    print('Betweenness centrality {}'.format(nx.betweenness_centrality(K)))
    '''Betweenness centrality of an edge e is the sum of the fraction of all-pairs shortest paths that pass through e:'''
    print('Edge betweenness centrality {}'.format(nx.edge_betweenness_centrality(G)))
    '''Current-flow closeness centrality is variant of closeness centrality based on effective resistance between nodes in a network.
    This metric is also known as information centrality'''
    print('Current flow closeness centrality {}'.format(nx.current_flow_closeness_centrality(K)))
    '''Current-flow betweenness centrality uses an electrical current model for information spreading in contrast to
    betweenness centrality which uses shortest paths.'''
    print('Current flow betweenness centrality {}'.format(nx.current_flow_betweenness_centrality(K)))
    print('Edge Current flow betweenness centrality {}'.format(nx.edge_current_flow_betweenness_centrality(K)))
    '''Eigenvector centrality computes the centrality for a node based on the centrality of its neighbors. '''
    GE = nx.path_graph(4)
    draw_graph(GE)
    centrality = nx.eigenvector_centrality(GE)
    print(['Eigenvector centrality for %s %0.2f' % (node, centrality[node]) for node in centrality])
    '''Katz centrality computes the centrality for a node based on the centrality of its neighbors. It is a generalization
    of the eigenvector centrality'''
    G = nx.path_graph(4)
    phi = (1 + math.sqrt(5)) / 2.0  # largest eigenvalue of adj matrix
    centrality = nx.katz_centrality(G, 1 / phi - 0.01)
    for n, c in sorted(centrality.items()):
        print("Katz centrality of %d %0.2f" % (n, c))
    '''Communicability'''
    '''The communicability between pairs of nodes in G is the sum of closed walks of different lengths starting at node
    u and ending at node v.'''
    GC = nx.Graph([(0, 1), (1, 2), (1, 5), (5, 4), (2, 4), (2, 3), (4, 3), (3, 6)])
    draw_graph(GC)
    print('Communicability {}'.format(nx.communicability(GC)))
    '''Communicability centrality, also called subgraph centrality, of a node n is the sum of closed walks of all lengths
    starting and ending at node n.'''
    print('Communicability centrality{}'.format(nx.communicability_centrality(GC)))
    '''Communicability betweenness measure makes use of the number of walks connecting every pair of nodes as the basis
    of a betweenness centrality measure.'''
    print('Communicability betveenness centrality{}'.format(nx.communicability_betweenness_centrality(GC)))
    '''The load centrality of a node is the fraction of all shortest paths that pass through that node'''
    print('Load centrality{}'.format(nx.load_centrality(GC)))
    '''A link between two actors (u and v) has a high dispersion when their mutual ties (s and t) are not well connected
    with each other.'''
    print('Dispersion {}'.format(nx.dispersion(GC, u=2, v=4)))


def chordal_alg():
    """ a chordal graph is one in which all cycles of four or more vertices have a chord, which is an edge that is not
    part of the cycle but connects two vertices of the cycle"""
    e = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    gc = nx.Graph(e)
    draw_graph(gc)
    print('Checks if the graph is chordal:? {}'.format(nx.is_chordal(gc)))
    gc.add_node(9)
    draw_graph(gc)
    print('Chordal graph cliques: {}'.format(nx.chordal_graph_cliques(gc)))
    # treewidth – The size of the largest clique in the graph minus one.
    print('Chordal graph treewidth: {}'.format(nx.chordal_graph_treewidth(gc)))
    # Returns the set of induced nodes in the path from s to t.
    print('Induced nodes in path: {}'.format(nx.find_induced_nodes(gc, 1, 5, 4)))


def cliques_alg2():
    """ cliques (subsets of vertices, all adjacent to each other, also called complete subgraphs"""
    e = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (5, 6)]
    gc = nx.Graph(e)
    draw_graph(gc)
    # list the cliques
    for clique in list(nx.enumerate_all_cliques(gc)):
        if len(clique) > 3:
            print("Clique: %s " % (clique))
    # Search for all maximal cliques in a graph.
    for clique in list(nx.find_cliques(gc)):
        print("Find clique: %s " % (clique))
    gcmax = nx.make_max_clique_graph(gc)
    for clique in list(gcmax):
        print("Make max clique: %s " % (clique))
    print("Max clique number: %d " % (nx.graph_clique_number(gc)))
    print("Number of cliques: %d " % (nx.graph_number_of_cliques(gc)))
    # Returns the size of the largest maximal clique containing each given node.
    for n, c in sorted(nx.node_clique_number(gc).items()):
        print("Size of the largest clique of node %d is %d " % (n, c))
    # Returns the number of maximal cliques for each node
    for n, c in sorted(nx.number_of_cliques(gc).items()):
        print("Number of maximal clique of node %d is %d " % (n, c))
    # Returns a list of cliques containing the given node.
    for n, c in list(nx.cliques_containing_node(gc, nodes=[1, 2, 5]).items()):
        print("Cliques containing node: %d is %s " % (n, c))


def clustering_alg():
    # Algorithms to characterize the number of triangles in a graph.
    g = nx.complete_graph(5)
    draw_graph(g)
    g.remove_edge(3, 4)
    print('Number of triangles that include the node as one vertex {}'.format(nx.triangles(g, 0)))
    for n, c in sorted(nx.triangles(g).items()):
        print("Number of triangles of node %d is %d " % (n, c))
    print('Number of triangles for all nodes {}'.format(nx.triangles(g)))
    print('Number of triangles for parameter nodes {}'.format(list(nx.triangles(g, (0, 1)).values())))
    """ Compute graph transitivity, the fraction of all possible triangles present in G.
    Possible triangles are identified by the number of “triads” (two edges with a shared vertex). """
    print('Transitivity {}'.format(nx.transitivity(g)))
    print('Clustering coeficient {}'.format(nx.clustering(g)))
    print('Average clustering coeficient of the graph {}'.format(nx.average_clustering(g)))
    print('Square clustering coeficient of the graph {}'.format(nx.square_clustering(g)))


def communities_alg():
    """A k-clique community is the union of all cliques of size k that can be reached through adjacent
    (sharing k-1 nodes) k-cliques"""
    g = nx.complete_graph(5)
    draw_graph(g)
    k5 = nx.convert_node_labels_to_integers(g, first_label=2)
    draw_graph(k5)
    g.add_edges_from(k5.edges())
    draw_graph(g)
    g.remove_edge(2, 4)
    draw_graph(g)
    print('k_clique_community {}'.format(list(nx.k_clique_communities(g, 4))))


def components_connectivity_alg():
    '''
    A graph is connected when there is a path between every pair of vertices. In a connected graph, there are no
    unreachable vertices
    '''
    g = nx.complete_graph(5)
    draw_graph(g)
    # g.remove_edges_from(ebunch=[(2,4), (3,4)])
    # draw_graph(g)
    print('Is connected ? {}'.format(nx.is_connected(g)))
    print('Nr of connected components: {}'.format(nx.number_connected_components(g)))
    G = nx.path_graph(4)
    draw_graph(G)
    G.add_path([10, 11, 12])
    draw_graph(G)
    print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
    print('Nr of connected components: {}'.format(nx.number_connected_components(G)))


def components_strong_connectivity_alg():
    '''
    It is strongly connected or strong if it contains a directed path from u to v and a
    directed path from v to u for every pair of vertices u, v
    '''
    g = nx.DiGraph()
    g.add_nodes_from(('a', 'b', 'c', 'd'))
    elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd')]
    g.add_edges_from(elist)
    draw_graph(g)
    print('Is strongly connected ? {}'.format(nx.is_strongly_connected(g)))

    g1 = nx.connected_caveman_graph(3, 4)
    # g1.add_nodes_from(('a', 'b', 'c', 'd'))
    # elist = [('a', 'b'), ('b', 'a'), ('a', 'c'), ('c', 'a'), ('c', 'd'), ('d', 'c'), ('b', 'd'), ('d', 'b')]
    # g1.add_edges_from(elist)
    draw_graph(g1)
    # print('Is strongly connected ? {}'.format(nx.is_strongly_connected(g1)))
    print('Number of strongly connected components {}'.format(nx.number_strongly_connected_components(g)))
    g2 = nx.cycle_graph(4, create_using=nx.DiGraph())
    g2.add_cycle([10, 11, 12])
    print("Strongly connected components are of lenghts {}".format(
        [len(c) for c in sorted(nx.strongly_connected_components(g2),
                                key=len, reverse=True)]))
    # generate subgraph
    g3 = nx.cycle_graph(4, create_using=nx.DiGraph())
    g3.add_cycle([10, 11, 12])
    draw_graph(g3)
    print([len(Gc) for Gc in sorted(nx.strongly_connected_component_subgraphs(g3), key=len, reverse=True)])

    # draw subgraphs - the strongly connected components of graph
    for gc in sorted(nx.strongly_connected_component_subgraphs(g3), key=len, reverse=True):
        draw_graph(gc)

    # draw subgraphs - the strongly connected components of graph
    print([len(gc) for gc in sorted(nx.kosaraju_strongly_connected_components(g3), key=len, reverse=True)])

    ''' The condensation of G is the graph with each of the strongly connected components contracted into a
    single node. '''
    print("condensation Graph ")
    draw_graph(nx.condensation(g3))


def components_weak_connectivity_alg():
    ''' A directed graph is weakly connected if, and only if, the graph is connected when the direction of the edge
    between nodes is ignored.'''
    g0 = nx.cycle_graph(4, create_using=nx.DiGraph())
    g0.add_cycle([10, 11, 12])
    g0.add_edges_from(ebunch=[(10, 1), (12, 2), (3, 10)])
    draw_graph(g0)
    print('Is weekly connected? {}'.format(nx.is_weakly_connected(g0)))
    print('Number of weekly connected components {}'.format(nx.number_weakly_connected_components(g0)))
    print("Weekly connected components are of lenghts {}".format(
        [len(c) for c in sorted(nx.weakly_connected_components(g0),
                                key=len, reverse=True)]))
    print("Weekly connected components' nodes: {}".format(
        [c for c in sorted(nx.weakly_connected_components(g0),
                           key=len, reverse=True)]))
    # generate subgraph and draws
    for gc in sorted(nx.weakly_connected_component_subgraphs(g0), key=len, reverse=True):
        draw_graph(gc)
    largest_cc = max(nx.weakly_connected_components(g0), key=len)
    print('Largest weekly connected component: {}'.format(largest_cc))


def component_attracting_alg():
    '''
    An attracting component in a directed graph is a strongly connected component with the property that a random walker
    on the graph will never leave the component, once it enters the component.
    The nodes in attracting components can also be thought of as recurrent nodes. If a random walker enters the
    attractor containing the node, then the node will be visited infinitely often.
    '''
    g0 = nx.cycle_graph(4, create_using=nx.DiGraph())
    g0.add_cycle([10, 11, 12])
    g0.add_edges_from(ebunch=[(10, 1), (12, 2), (3, 10)])
    draw_graph(g0)
    print('Is attracting component: {}'.format(nx.is_attracting_component(g0)))
    print('Number of attracting components: {}'.format(nx.number_attracting_components(g0)))
    print("Attracting components': {}".format(
        [c for c in sorted(nx.attracting_components(g0),
                           key=len, reverse=True)]))
    # generate subgraph and draws
    for gc in (nx.attracting_component_subgraphs(g0)):
        draw_graph(gc)


def component_biconnected_alg():
    '''
    A graph is biconnected if, and only if, it cannot be disconnected by removing only one node (and all edges incident
     on that node). If removing a node increases the number of disconnected components in the graph,
    that node is called an articulation point, or cut vertex. A biconnected graph has no articulation points.
    '''
    g = nx.lollipop_graph(5, 1)
    draw_graph(g)
    print('Is beconnected component: {}'.format(nx.is_biconnected(g)))
    bicomponents = list(nx.biconnected_components(g))
    print('Beconnected component: {}'.format(list(nx.biconnected_components(g))))
    len(bicomponents)
    g.add_edge(0, 5)
    draw_graph(g)
    print('Is beconnected component: {}'.format(nx.is_biconnected(g)))
    bicomponents = list(nx.biconnected_components(g))
    len(bicomponents)
    print("Beconnected components': {}".format(
        [len(c) for c in sorted(nx.biconnected_components(g),
                                key=len, reverse=True)]))
    g = nx.barbell_graph(4, 2)
    draw_graph(g)
    print('Is beconnected component: {}'.format(nx.is_biconnected(g)))
    bicomponents = list(nx.biconnected_components(g))
    len(bicomponents)
    print("Beconnected components': {}".format(
        [len(c) for c in sorted(nx.biconnected_components(g),
                                key=len, reverse=True)]))
    # articulation point
    print('Articulation point: {}'.format(len(list(nx.articulation_points(g)))))
    g.add_edge(2, 8)
    print('Is beconnected component: {}'.format(nx.is_biconnected(g)))
    print('Articulation point: {}'.format(len(list(nx.articulation_points(g)))))


def component_semiconnectedness_alg():
    '''
    A graph is semiconnected if, and only if, for any pair of nodes, either one is reachable from the other,
    or they are mutually reachable.
    '''
    g = nx.path_graph(4, create_using=nx.DiGraph())
    draw_graph(g)
    print('Is semiconnected component: {}'.format(nx.nx.is_semiconnected(g)))
    g = nx.DiGraph([(1, 2), (3, 2)])
    draw_graph(g)
    print('Is semiconnected component: {}'.format(nx.nx.is_semiconnected(g)))


def connectivity_k_node_components_alg():
    '''
    A k-component is a maximal subgraph of a graph G that has, at least, node connectivity k: we need to remove at
    least k nodes to break it into more components
    '''
    g = nx.petersen_graph()
    draw_graph(g)
    print('k-components: {}'.format(nx.k_components(g)))
    g.remove_edges_from(ebunch=[(4, 5), (7, 9)])
    draw_graph(g)
    print('k-components: {}'.format(nx.k_components(g)))


def connectivity_node_cuts():
    '''
     ie the set (or sets) of nodes of cardinality equal to the node connectivity of G. Thus if removed, would break G
     into two or more connected components.
    '''
    g = nx.grid_2d_graph(5, 5)
    draw_graph(g)
    cutsets = list(nx.all_node_cuts(g))
    print('Cutsets: {}'.format(cutsets))

    len(cutsets)
    # True, if every element's len is 2
    all(2 == len(cutset) for cutset in cutsets)
    nx.node_connectivity(g)


def connectivity_flow_based_alg():
    g = nx.grid_2d_graph(5, 5)
    draw_graph(g)
    print('Average node connectivity: {}'.format(nx.average_node_connectivity(g)))
    g = nx.petersen_graph()
    g.remove_edges_from(ebunch=[(4, 5), (7, 9), (3, 4), (1, 2)])
    draw_graph(g)
    print('Average node connectivity: {}'.format(nx.average_node_connectivity(g)))
    print('All pairs of node connectivity: {}'.format(nx.all_pairs_node_connectivity(g, nbunch=(6,))))
    '''The edge connectivity is equal to the minimum number of edges that must be removed to disconnect G or render it
    trivial. If source and target nodes are provided, this function returns the local edge connectivity: the minimum
    number of edges that must be removed to break all paths from source to target in G.'''
    g1 = nx.icosahedral_graph()
    draw_graph(g1)
    g1.remove_edges_from(ebunch=[(2, 7), (3, 8)])
    print('Edge connectivity: {}'.format(nx.edge_connectivity(g1, s=2, t=5)))
    print('Node connectivity: {}'.format(nx.node_connectivity(g1)))
    '''Returns a set of edges of minimum cardinality that disconnects G.'''
    print('Minimum edge cut: {}'.format(nx.minimum_edge_cut(g1)))
    print('Minimum node cut: {}'.format(nx.minimum_node_cut(g1)))
    print('Minimum source traget edge cut: {}'.format(minimum_st_edge_cut(g1, s=3, t=5)))
    print('Minimum source target node cut: {}'.format(minimum_st_node_cut(g1, s=9, t=11)))
    '''Determine the minimum edge cut of a connected graph using the Stoer-Wagner algorithm. In weighted cases, all
    weights must be nonnegative.'''
    G = nx.Graph()
    G.add_edge('x', 'a', weight=3)
    G.add_edge('x', 'b', weight=1)
    G.add_edge('a', 'c', weight=3)
    G.add_edge('b', 'c', weight=5)
    G.add_edge('b', 'd', weight=4)
    G.add_edge('d', 'e', weight=2)
    G.add_edge('c', 'y', weight=2)
    G.add_edge('e', 'y', weight=3)
    draw_graph(G)
    cut_value, partition = nx.stoer_wagner(G)
    print('Cut value: {} and partition {}'.format(cut_value, partition))


def core_alg():
    '''A k-core is a maximal subgraph that contains nodes of degree k or more.
    The k-core is found by recursively pruning nodes with degrees less than k.'''
    g0 = nx.cycle_graph(4, create_using=nx.DiGraph())
    g0.add_cycle([10, 11, 12])
    g0.add_edges_from(ebunch=[(10, 1), (12, 2), (3, 10)])
    g0.remove_edges_from(ebunch=[(2, 12), (10, 12), (5, 12), (6, 12), (7, 12), (8, 12)])
    draw_graph(g0)
    print('k-core value: {}'.format(nx.core_number(g0)))
    print('k-core value: {}'.format(nx.k_core(g0)))
    g = nx.DiGraph()
    g.add_nodes_from(('a', 'b', 'c', 'd'))
    elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd')]
    g.add_edges_from(elist)
    draw_graph(g)
    print('k-core value: {}'.format(nx.core_number(g)))
    print('k-core value: {}'.format(nx.k_core(g)))
    '''A k-core is a maximal subgraph that contains nodes of degree k or more.'''
    draw_graph(nx.k_core(g))
    '''The k-shell is the subgraph of nodes in the k-core but not in the (k+1)-core.'''
    draw_graph(nx.k_shell(g))
    '''he k-crust is the graph G with the k-core removed.'''
    draw_graph(nx.k_crust(g))
    '''The k-corona is the subgraph of nodes in the k-core which have exactly k neighbours in the k-core.'''
    draw_graph(nx.k_corona(g, k=2))
    print('k-core value: {}'.format(nx.core_number(g0)))
    draw_graph(nx.k_shell(g0))


def cycles_alg():
    '''Returns a list of cycles which form a basis for cycles of G.'''
    g = nx.Graph()
    g.add_cycle([0, 1, 2, 3])
    g.add_cycle([0, 3, 4, 5])
    draw_graph(g)
    print('Cycle basis: {}'.format(nx.cycle_basis(g, 0)))
    g0 = nx.DiGraph([(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)])
    print('Simple cycles: {}'.format(list(nx.simple_cycles(g0))))
    copyg0 = g0.copy()
    copyg0.remove_nodes_from([1])
    copyg0.remove_edges_from([(0, 1)])
    print('Simple cycles after removeing nodes and edges: {}'.format(list(nx.simple_cycles(copyg0))))
    draw_graph(copyg0)
    print('Find cycles')
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    try:
        nx.find_cycle(G, orientation='original')
    except:
        pass
    print('Find cycles ignore orientation {}'.format(list(nx.find_cycle(G, orientation='ignore'))))


def directed_acyclic_graphs_alg():
    '''Return all nodes having a path to source in G.'''
    g = nx.DiGraph()
    g.add_nodes_from(('a', 'b', 'c', 'd'))
    elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd')]
    g.add_edges_from(elist)
    print('Return all nodes having a path to source in G: {}'.format(list(nx.ancestors(g, 'd'))))
    print('Return all nodes reachable from source in G: {}'.format(list(nx.descendants(g, 'a'))))
    '''A topological sort is a nonunique permutation of the nodes such that an edge from u to v implies that u appears
    before v in the topological sort order.'''
    print('Return a list of nodes in topological sort order in G: {}'.format(list(nx.topological_sort(g))))
    print('Is G directed graph: {}'.format(nx.is_directed(g)))
    '''A directed graph is aperiodic if there is no integer k > 1 that divides the length of every cycle in the graph.'''
    print('Is G aperiodic: {}'.format(nx.is_aperiodic(g)))
    '''The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that for all v,w in V there is an edge (v,w)
    in E+ if and only if there is a non-null path from v to w in G.'''
    draw_graph(g)
    draw_graph(nx.transitive_closure(g))
    '''An antichain is a subset of a partially ordered set such that any two elements in the subset are incomparable.'''
    print('Generates antichains from a DAG: {}'.format(list(nx.antichains(g))))
    print('Returns the longest path in a DAG: {}'.format(list(nx.dag_longest_path(g))))
    print('Returns the longest path lenght in a DAG: {}'.format(nx.dag_longest_path_length(g)))


def distance_measures_alg():
    g = nx.DiGraph()
    g.add_nodes_from(('a', 'b', 'c', 'd'))
    elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd'), ('d', 'a')]
    g.add_edges_from(elist)
    '''The center is the set of nodes with eccentricity equal to radius.'''
    print('Returns the center of a G: {}'.format(list(nx.center(g))))
    '''The diameter is the maximum eccentricity.'''
    print('Returns the diameter of a G: {}'.format(nx.diameter(g)))
    '''The eccentricity of a node v is the maximum distance from v to all other nodes in G'''
    print('Returns the eccentricity of a G: {}'.format(nx.eccentricity(g)))
    '''The periphery is the set of nodes with eccentricity equal to the diameter.'''
    print('Returns the periphery of a G: {}'.format(nx.periphery(g)))
    '''The radius is the minimum eccentricity.'''
    print('Returns the radius of a G: {}'.format(nx.radius(g)))
    draw_graph(g)


def distance_regular_graphs_alg():
    '''A connected graph G is distance-regular if for any nodes x,y and any integers i,j=0,1,...,d (where d is the
    graph diameter), the number of vertices at distance i from x and distance j
    from y depends only on i,j and the graph distance between x and y, independently of the choice of x and y'''
    g = nx.hypercube_graph(6)
    g1 = nx.dodecahedral_graph()
    print('Returns if distance regular graph G: {}'.format(nx.is_distance_regular(g)))
    '''Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d such that for any 2 vertices x,y in G at a
    distance i=d(x,y), there are exactly c_i neighbors of y at a distance of i-1 from x and b_i neighbors of y at a
    distance of i+1 from x'''
    print('Returns intersection array of graph G: {}'.format(list(nx.intersection_array(g))))
    b, c = nx.intersection_array(g)
    print('Returns iglobal parameters of intersection array: {}'.format(list(nx.global_parameters(b, c))))
    draw_graph(g)
    g = nx.Graph()
    g.add_nodes_from(('a', 'b', 'c', 'd'))
    elist = [('a', 'b'), ('b', 'c'), ('a', 'c'), ('c', 'd'), ('d', 'a')]
    g.add_edges_from(elist)
    print('Returns if distance regular graph G: {}'.format(nx.is_distance_regular(g)))
    draw_graph(g)


def dominators_alg():
    ''' a node d dominates a node n if every path from the entry node to n must go through d.'''
    g = nx.DiGraph([(1, 2), (1, 3), (2, 5), (3, 4), (4, 5)])
    print('Returns the immediate dominators of all nodes of a directed graph.: {}'.format(
        sorted(nx.immediate_dominators(g, 1).items())))
    print('Returns the dominance frontiers of all nodes of a directed graph G: {}'
          .format(sorted((u, sorted(df)) for u, df in nx.dominance_frontiers(g, 1).items())))
    '''A dominating set for a graph G = (V, E) is a node subset D of V such that every node not in D is
    adjacent to at least one member of D'''
    print('Returns the dominating set: {}'.format(list(nx.dominating_set(g, 1))))
    print('Returns if the given node set is dominating: {}'.format(nx.is_dominating_set(g, nbunch=[1, 4])))
    draw_graph(g)


def eulerian_alg():
    ''''''
    g = nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]})
    print('Returns True if Eulerian cycle: {}'.format(nx.is_eulerian(g)))
    draw_graph(g)
    print('Returns True if Eulerian cycle: {}'.format(nx.is_eulerian(nx.complete_graph(5))))
    draw_graph(nx.complete_graph(5))
    print('Returns True if Eulerian cycle {}'.format(nx.is_eulerian(nx.petersen_graph())))
    draw_graph(nx.petersen_graph())
    '''An Eulerian circuit is a path that crosses every edge in G exactly once and finishes at the starting node.'''
    print('Returns eulerian circuit: {}'.format(list(nx.eulerian_circuit(nx.complete_graph(5), source=1))))


def flow_alg():
    ''''''
    G = nx.DiGraph()
    G.add_edge('x', 'a', capacity=3.0)
    G.add_edge('x', 'b', capacity=1.0)
    G.add_edge('a', 'c', capacity=3.0)
    G.add_edge('b', 'c', capacity=5.0)
    G.add_edge('b', 'd', capacity=4.0)
    G.add_edge('d', 'e', capacity=2.0)
    G.add_edge('c', 'y', capacity=2.0)
    G.add_edge('e', 'y', capacity=3.0)
    # maximum flow, i.e., net outflow from the source and a dictionary containing the value of the flow that went
    # through each edge..
    flow_value, flow_dict = nx.maximum_flow(G, 'x', 'y')
    print('Net flow from source to target: {}'.format(flow_value))
    print('Capacities through edges: {}'.format(list(flow_dict)))
    print(flow_dict['x']['b'])
    print('Maximum flow value: {}'.format(nx.maximum_flow(G, 'x', 'y', capacity='capacity')))
    # minimum cut
    cut_value, partition = nx.minimum_cut(G, 'x', 'y')
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    print(sorted(cutset))
    cut_value == sum(G.edge[u][v]['capacity'] for (u, v) in cutset)
    print('Cut value {}'.format(cut_value))
    draw_graph(G)
    # network simplex
    G1 = nx.DiGraph()
    G1.add_node('a', demand=-5)
    G1.add_node('d', demand=5)
    G1.add_edge('a', 'b', weight=3, capacity=4)
    G1.add_edge('a', 'c', weight=6, capacity=10)
    G1.add_edge('b', 'd', weight=1, capacity=9)
    G1.add_edge('c', 'd', weight=2, capacity=5)
    flowCost, flowDict = nx.network_simplex(G1)
    print('Network-simplex - flowcost {}'.format(flowCost))
    print('Network-simplex - flowDict {}'.format(list(flowDict)))
    draw_graph(G1)


def graphical_degree_seq_alg():
    '''A graphical degree sequence is valid if some graph can realize it.'''
    G = nx.path_graph(4)
    sequence = G.degree().values()
    outsequence = sequence
    print('Is sequence is valid for Graph?  {}'.format(nx.is_graphical(sequence)))
    print('Is sequence is valid for DiGraph?  {}'.format(nx.is_digraphical(sequence, outsequence)))
    # multigraph can realize the sequence
    print('Is sequence is valid for MultiGraph?  {}'.format(nx.is_multigraphical(sequence)))
    # pseudograph can realize the sequence
    print('Is sequence valid for PseudoGraph?  {}'.format(nx.is_multigraphical(sequence)))
    # Havel-Hakimi
    print('Is sequence valid for Graph :: Erdos-Gallai?  {}'.format(nx.is_valid_degree_sequence_erdos_gallai(sequence)))
    # Erdos-Gallai
    print('Is sequence valid for Graph :: Havel-Hakimi?  {}'.format(nx.is_valid_degree_sequence_havel_hakimi(sequence)))
    draw_graph(G)


def hierarchy_alg():
    '''Flow hierarchy is defined as the fraction of edges not participating in cycles in a directed graph'''
    G1 = nx.DiGraph()
    G1.add_node('a', demand=-5)
    G1.add_node('d', demand=5)
    G1.add_edge('a', 'b', weight=3, capacity=4)
    G1.add_edge('a', 'c', weight=6, capacity=10)
    G1.add_edge('b', 'd', weight=1, capacity=9)
    G1.add_edge('c', 'd', weight=2, capacity=5)
    G1.add_cycle(['a', 'b'])
    print('Flow hierarchy ::   {}'.format(nx.flow_hierarchy(G1)))
    draw_graph(G1)


def hybrid_alg():
    '''A graph is locally (k, l)-connected if for each edge (u, v) in the graph there are at least l edge-disjoint
    paths of length at most k joining u to v.'''
    g1 = nx.dodecahedral_graph()
    print('Is k,l connected ::   {}'.format(nx.is_kl_connected(g1, k=5, l=3)))
    draw_graph(g1)
    draw_graph(nx.kl_connected_subgraph(g1, k=5, l=3))


def isolates_alg():
    '''Determine of node n is an isolate (degree zero).'''
    G = nx.Graph()
    G.add_edge(1, 2)
    G.add_node(3)
    print('Is {} isolated ::   {}'.format(2, nx.is_isolate(G, 2)))
    print('Is {} isolated ::   {}'.format(3, nx.is_isolate(G, 3)))
    draw_graph(G)


def isomorfism_alg():
    '''Two graphs which contain the same number of graph vertices connected in the same way are said to be isomorphic'''
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    G1.add_path([1, 2, 3, 4], weight=1)
    G2.add_path([10, 20, 30, 40], weight=2)
    print('Is {} and {} isomorfic::   {}'.format('G1', 'G2', nx.is_isomorphic(G1, G2)))
    # no weights considered
    em = iso.numerical_edge_match('weight', 1)
    nx.is_isomorphic(G1, G2, edge_match=em)  # match weights
    print('Is {} and {} isomorfic::   {}'.format('G1', 'G2', nx.is_isomorphic(G1, G2, edge_match=em)))
    draw_graph(G1)
    draw_graph(G2)
    # isomorfism for multigraphs
    G1 = nx.MultiDiGraph()
    G2 = nx.MultiDiGraph()
    G1.add_nodes_from([1, 2, 3], fill='red')
    G2.add_nodes_from([10, 20, 30, 40], fill='red')
    G1.add_path([1, 2, 3, 4], weight=3, linewidth=2.5)
    G2.add_path([10, 20, 30, 40], weight=3)
    nm = iso.categorical_node_match('fill', 'red')
    print('Is {} and {} isomorfic categorical ::   {}'.format('G1', 'G2', nx.is_isomorphic(G1, G2, edge_match=em)))
    draw_graph(G1)
    draw_graph(G2)
    # For multidigraphs G1 and G2, using ‘weight’ edge attribute (default: 7)
    G1.add_edge(1, 2, weight=7)
    G2.add_edge(10, 20)
    em = iso.numerical_multiedge_match('weight', 7, rtol=1e-6)
    print('Is {} and {} isomorfic categorical ::   {}'.format('G1', 'G2', nx.is_isomorphic(G1, G2, edge_match=em)))
    draw_graph(G1)
    draw_graph(G2)
    '''For multigraphs G1 and G2, using ‘weight’ and ‘linewidth’ edge attributes with default values 7 and 2.5. Also
    using ‘fill’ node attribute with default value ‘red’.'''
    em = iso.numerical_multiedge_match(['weight', 'linewidth'], [7, 2.5])
    nm = iso.categorical_node_match('fill', 'red')
    print('Is {} and {} isomorfic categorical ::   {}'.format('G1', 'G2',
                                                              nx.is_isomorphic(G1, G2, edge_match=em, node_match=nm)))
    g1 = nx.path_graph(4)
    g2 = nx.path_graph(5)
    print('Could {} and {} be isomorfic ::   {}'.format('G1', 'G2', nx.could_be_isomorphic(g1, g2)))
    print('Fast could {} and {} be isomorfic ::   {}'.format('G1', 'G2', nx.fast_could_be_isomorphic(g1, g2)))
    draw_graph(g1)
    draw_graph(g2)


def pagerank_alg():
    G = nx.DiGraph(nx.path_graph(4))
    print('Pagerank for {} ::   {}'.format('G', nx.pagerank(G, alpha=0.9)))
    draw_graph(G)


def hits_alg():
    '''Hubs and authorities analysis of graph structure. The HITS algorithm computes two numbers for a node. Authorities
    estimates the node value based on the incoming links. Hubs estimates the node value based on outgoing links.'''
    G = nx.path_graph(4)
    h, a = nx.hits(G)
    print('HITs for {} ::   {}'.format('G', list(nx.hits(G))))
    print('HUB matrix for {} ::   {}'.format('G', list(nx.hub_matrix(G))))
    print('Authority matrix for {} ::   {}'.format('G', list(nx.authority_matrix(G))))
    draw_graph(G)


def link_prediction_alg():
    '''Compute the resource allocation index of all node pairs in ebunch.'''
    G = nx.complete_graph(5)
    preds = nx.resource_allocation_index(G, [(0, 1), (2, 3), (2, 0)])
    for u, v, p in preds:
        print('Resource allocation index for edge: (%d, %d) -> %.8f' % (u, v, p))
    jaccard_coefficient = nx.jaccard_coefficient(G, [(0, 1), (2, 3), (2, 0)])
    for u, v, j in jaccard_coefficient:
        print('Jaccard coefficient for node pairs: (%d, %d) -> %.8f' % (u, v, j))
    adamic_adar_index = nx.jaccard_coefficient(G, [(0, 1), (2, 3), (2, 0)])
    for u, v, a in adamic_adar_index:
        print('Adamic-Adar index for node pairs: (%d, %d) -> %.8f' % (u, v, a))
    '''Compute the preferential attachment score of all node pairs in ebunch'''
    preferencial_attachment_score = nx.preferential_attachment(G, [(0, 1), (2, 3), (2, 0)])
    for u, v, pa in preferencial_attachment_score:
        print('Preferencial attachment score for node pairs: (%d, %d) -> %.8f' % (u, v, pa))
    '''Count number Soundarajan Hopcroft '''
    G1 = nx.path_graph(4)
    G1.node[0]['community'] = 0
    G1.node[1]['community'] = 0
    G1.node[2]['community'] = 0
    G1.node[3]['community'] = 0
    cn_soundarajan_hopcroft = nx.cn_soundarajan_hopcroft(G1, [(0, 2), (2, 3), (2, 0)], community='community')
    for u, v, sh in cn_soundarajan_hopcroft:
        print('Count number Soundarajan Hopcroft for node pairs: (%d, %d) -> %.8f' % (u, v, sh))
    '''Compute the resource allocation index of all node pairs in ebunch using community information.'''
    G2 = nx.Graph()
    G2.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    G2.node[0]['community'] = 0
    G2.node[1]['community'] = 0
    G2.node[2]['community'] = 1
    G2.node[3]['community'] = 0
    ra_index_soundarajan_hopcroft = nx.ra_index_soundarajan_hopcroft(G2, [(0, 3)])
    for u, v, ra in ra_index_soundarajan_hopcroft:
        print('Resource allocation index Soundarajan Hopcroft for node pairs: (%d, %d) -> %.8f' % (u, v, ra))
    '''Compute the ratio of within- and inter-cluster common neighbors of all node pairs in ebunch.
    For two nodes u and v, if a common neighbor w belongs to the same community as them, w is considered as
    within-cluster common neighbor of u and v. Otherwise, it is considered as inter-cluster common neighbor of u and v.
    The ratio between the size of the set of within- and inter-cluster common neighbors is defined as the WIC measure.
    '''
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    G3.node[0]['community'] = 0
    G3.node[1]['community'] = 1
    G3.node[2]['community'] = 0
    G3.node[3]['community'] = 0
    G3.node[4]['community'] = 0
    wic_measure = nx.within_inter_cluster(G3, [(0, 4)], delta=0.5)
    for u, v, wic in wic_measure:
        print('WIC measure for node pairs: (%d, %d) -> %.8f' % (u, v, wic))
    draw_graph(G3)


def matching_alg():
    '''A matching is a subset of edges in which no node occurs more than once. The cardinality of a matching is the
    number of matched edges.'''
    g = nx.karate_club_graph()
    print('Maximal matching: {}'.format(list(nx.maximal_matching(g))))
    draw_graph(g)
    '''A matching is a subset of edges in which no node occurs more than once. The cardinality of a matching is the
    number of matched edges. The weight of a matching is the sum of the weights of its edges.'''
    G3 = nx.Graph()
    G3.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 4), (2, 4), (3, 4)])
    G3.node[0]['weight'] = 2
    G3.node[1]['weight'] = 1
    G3.node[2]['weight'] = 3
    G3.node[3]['weight'] = 4
    G3.node[4]['weight'] = 5
    print('Max weight matching: {}'.format(list(nx.max_weight_matching(G3))))
    draw_graph(G3)


def minors_alg():
    '''Edge contraction identifies the two endpoints of the edge as a single node incident to any edge that was incident
     to the original two nodes. A graph that results from edge contraction is called a minor of the original graph.'''
    G = nx.cycle_graph(4)
    nx.contracted_edge(G, (2, 3))
    draw_graph(G)
    draw_graph(nx.contracted_edge(G, (2, 3)))
    C5 = nx.cycle_graph(5)
    C4 = nx.cycle_graph(4)
    M = nx.contracted_edge(C5, (0, 1), self_loops=False)
    print('Are {} and {} isomorphic? :: {} '.format('M', 'C4', nx.is_isomorphic(M, C4)))
    draw_graph(C4)
    draw_graph(M)
    '''Node contraction identifies the two nodes as a single node incident to any edge that was incident to the original
     two nodes.'''
    G = nx.cycle_graph(4)
    M = nx.contracted_nodes(G, 1, 3)
    P3 = nx.path_graph(3)
    nx.is_isomorphic(M, P3)
    print('Are {} and {} isomorphic? :: {} '.format('M', 'P3', nx.is_isomorphic(M, P3)))
    draw_graph(M)
    draw_graph(P3)
    G = nx.cycle_graph(4)
    M = nx.contracted_nodes(G, 1, 3)
    P3 = nx.path_graph(3)
    print('Are {} and {} isomorphic? :: {} '.format('M', 'P3', nx.is_isomorphic(M, P3)))
    '''Returns the quotient graph of G under the specified equivalence relation on nodes.'''
    G = nx.complete_bipartite_graph(2, 3)
    same_neighbors = lambda u, v: (u not in G[v] and v not in G[u] and G[u] == G[v])
    Q = nx.quotient_graph(G, same_neighbors)
    K2 = nx.complete_graph(2)
    print('Are {} and {} isomorphic? :: {} '.format('Q', 'K2', nx.is_isomorphic(Q, K2)))
    draw_graph(Q)
    draw_graph(K2)
    G = nx.DiGraph()
    edges = ['ab', 'be', 'bf', 'bc', 'cg', 'cd', 'dc', 'dh', 'ea', 'ef', 'fg', 'gf', 'hd', 'hf']
    G.add_edges_from(tuple(x) for x in edges)
    components = list(nx.strongly_connected_components(G))
    print('Components: {}'.format(sorted(sorted(component) for component in components)))
    C = nx.condensation(G, components)
    component_of = C.graph['mapping']
    same_component = lambda u, v: component_of[u] == component_of[v]
    Q = nx.quotient_graph(G, same_component)
    print('Components {} and {} are isomorfic: {}'.format('C', 'Q', nx.is_isomorphic(C, Q)))
    draw_graph(G)
    draw_graph(C)
    draw_graph(Q)
    #
    K24 = nx.complete_bipartite_graph(2, 4)
    K34 = nx.complete_bipartite_graph(3, 4)
    C = nx.contracted_nodes(K34, 1, 2)
    nodes = {1, 2}
    is_contracted = lambda u, v: u in nodes and v in nodes
    Q = nx.quotient_graph(K34, is_contracted)
    print('Components {} and {} are isomorfic: {}'.format('Q', 'C', nx.is_isomorphic(Q, C)))
    print('Components {} and {} are isomorfic: {}'.format('Q', 'K24', nx.is_isomorphic(Q, K24)))
    draw_graph(K34)
    draw_graph(Q)
    draw_graph(C)
    draw_graph(K24)


def maximal_independent_set_alg():
    '''An independent set is a set of nodes such that the subgraph of G induced by these nodes contains no edges. '''
    G = nx.path_graph(5)
    print('Maximal independent set: {} that must contain nodes: {}'.format(nx.maximal_independent_set(G, nodes=[1]), 1))
    draw_graph(G)


def minimum_spanning_tree_alg():
    '''A minimum spanning tree is a subgraph of the graph (a tree) with the minimum sum of edge weights. If the graph is
     not connected a spanning forest is constructed. A spanning forest is a union of the spanning trees for each
     connected component of the graph.'''
    G = nx.cycle_graph(4)
    G.add_edge(0, 3, weight=2)
    T = nx.minimum_spanning_tree(G)
    print('Minimum spanning tree: {}'.format(sorted(T.edges(data=True))))
    draw_graph(G)
    draw_graph(T)
    mst = nx.minimum_spanning_edges(G, data=False)  # a generator of MST edges
    edgelist = list(mst)  # make a list of the edges
    print('Minimum spanning edges: {}'.format(sorted(edgelist)))


def operators_alg():
    '''graph operations'''
    G = nx.cycle_graph(4)
    # DG = nx.DiGraph(nx.path_graph(4))
    G1 = nx.DiGraph()
    G1.add_nodes_from([1, 2, 3])
    G1.add_path([1, 2, 3, 4])
    G2 = nx.karate_club_graph()
    G3 = nx.DiGraph()
    G3.add_nodes_from([6, 7, 8])
    G3.add_edges_from(ebunch=[(6, 7), (8, 6)])
    G4 = nx.DiGraph()
    G4.add_nodes_from([6, 7, 8, 4])
    G4.add_edges_from(ebunch=[(6, 7), (8, 6), (6, 4)])
    G5 = nx.path_graph(4)
    # G5.remove_node(4)

    print('Complement graph')
    '''complement or inverse of a graph G is a graph H on the same vertices such that two distinct vertices of H are
    adjacent if and only if they are not adjacent in G'''
    T = nx.complement(G)
    draw_graph(G)
    draw_graph(T)
    '''the converse, transpose or reverse of a directed graph G is another directed graph on the same set of vertices
    with all of the edges reversed compared to the orientation of the corresponding edges in G'''
    print('Reverse graph')
    draw_graph(G1)
    R = nx.reverse(G1)
    draw_graph(R)
    '''Composition is the simple union of the node sets and edge sets. The node sets of G and H do not need to be
    isjoint.'''
    print('Compose new graph from two graphs')
    C = nx.compose(G2, G1)
    draw_graph(G2)
    draw_graph(C)
    '''Union of graphs G and H'''
    print('Union of graphs G and H')
    U = nx.union(G1, G3)
    draw_graph(G1)
    draw_graph(G3)
    draw_graph(U)
    '''Disjoint union'''
    print('Disjoint union of two graphs')
    DU = nx.disjoint_union(G1, G4)
    draw_graph(G1)
    draw_graph(G4)
    draw_graph(DU)
    ''' intersection is a new graph that contains only the edges that exist in both G and H'''
    print('Intersection')
    G5 = nx.path_graph(3)
    H5 = nx.path_graph(5)
    R5 = G.copy()
    R5.remove_nodes_from(n for n in G5 if n not in H5)
    draw_graph(R5)
    '''difference - Return a new graph that contains the edges that exist in G but not in H.'''
    print('Difference of two graphs')
    # D = nx.difference(G1, G5)
    draw_graph(G1)
    draw_graph(G5)
    # draw_graph(D)
    print('Cartesian product')
    H = nx.Graph()
    G.add_node(0, a1=True)
    H.add_node('a', a2='Spam')
    P = nx.cartesian_product(G, H)
    draw_graph(G)
    draw_graph(H)
    draw_graph(P)
    '''The lexicographical product P of the graphs G and H has a node set that is the Cartesian product of the node
    sets, $V(P)=V(G) imes V(H)$. P has an edge ((u,v),(x,y)) if and only if (u,v) is an edge in G or u==v and (x,y) is
    an edge in H.'''
    print('Lexicgraphic product')
    G = nx.Graph()
    H = nx.Graph()
    G.add_node(0, a1=True)
    H.add_node('a', a2='Spam')
    P = nx.lexicographic_product(G, H)
    draw_graph(G)
    draw_graph(H)
    draw_graph(P)
    '''The strong product P of the graphs G and H has a node set that is the Cartesian product of the node sets,
    $V(P)=V(G) imes V(H)$. P has an edge ((u,v),(x,y)) if and only if u==v and (x,y) is an edge in H, or x==y and
    (u,v) is an edge in G, or (u,v) is an edge in G and (x,y) is an edge in H.'''
    print('Strong product')
    G = nx.Graph()
    H = nx.Graph()
    G.add_node(0, a1=True)
    H.add_node('a', a2='Spam')
    P = nx.strong_product(G, H)
    draw_graph(G)
    draw_graph(H)
    draw_graph(P)
    '''The tensor product P of the graphs G and H has a node set that is the Cartesian product of the node sets,
    V(P)=V(G) \times V(H). P has an edge ((u,v),(x,y)) if and only if (u,x) is an edge in G and (v,y) is an
    edge in H.'''
    print('Tensor product')
    G = nx.Graph()
    H = nx.Graph()
    G.add_node(0, a1=True)
    H.add_node('a', a2='Spam')
    P = nx.tensor_product(G, H)
    draw_graph(P)
    '''The k-th power of a simple graph G = (V, E) is the graph G^k whose vertex set is V, two distinct vertices u,v
    are adjacent in G^k if and only if the shortest path distance between u and v in G is at most k.'''
    print('Power')
    G = nx.cycle_graph(5)
    draw_graph(G)
    draw_graph(nx.power(G, 2))
    G = nx.cycle_graph(8)
    draw_graph(G)
    draw_graph(nx.power(G, 4))


def rich_club_coefficient_alg():
    '''The rich-club coefficient is the ratio, for every degree k, of the number of actual to the number of potential
    edges for nodes with degree greater than k'''
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    print('Rich club coefficient values: {}'.format(list(nx.rich_club_coefficient(G, normalized=False))))


def shortest_path_alg():
    '''shortest path algoritms'''
    G = nx.path_graph(5)
    print(nx.shortest_path(G, source=1, target=3))
    p = nx.shortest_path(G, source=0)  # target not specified
    print(p[4])
    p = nx.shortest_path(G, target=4)  # source not specified
    print([0])
    p = nx.shortest_path(G)  # source,target not specified
    print(p[0][4])
    G = nx.Graph()
    G.add_path([0, 1, 2])
    G.add_path([0, 10, 2])
    draw_graph(G)
    print(nx.shortest_path_length(G, source=0, target=2))
    print([p for p in nx.all_shortest_paths(G, source=0, target=2)])
    print([(p, len(p) - 1) for p in nx.all_shortest_paths(G, source=0, target=2)])
    print('Average shortest path: {}'.format(nx.average_shortest_path_length(G)))
    print('Has path? {} Path lenght: {}'.format(nx.has_path(G, source=1, target=10),
                                                nx.shortest_path_length(G, source=1, target=10)))


def shortest_path_advanced_alg():
    '''Dijkstra algorithm'''
    G = nx.path_graph(5)
    draw_graph(G)
    print('Single source shortest path: {} '.format(list(nx.single_source_shortest_path(G, source=3))))
    print('Single source shortest path lenghts: {}'.format(nx.single_source_shortest_path(G, source=3)))
    print('All pairs shortest path: {}'.format(nx.all_pairs_shortest_path(G)))
    print('Predecessors: {}'.format(nx.predecessor(G, source=0)))

    '''alg. for weigthed graphs'''
    print('Dijkstra alg: {}'.format(list(nx.dijkstra_path(G, 0, 4))))
    print('Dijkstra path lenght: {}'.format(nx.dijkstra_path_length(G, 0, 3)))
    print('Single source Dijkstra path: {}'.format(nx.single_source_dijkstra_path(G, 0, 3)))
    print('Single source Dijkstra path lenght: {}'.format(nx.single_source_dijkstra_path_length(G, source=0)))
    print('All pairs Dijkstra path: {}'.format(nx.all_pairs_dijkstra_path(G)))
    print('All pairs Dijkstra path lenght: {}'.format(nx.all_pairs_dijkstra_path_length(G)))
    length, path = nx.single_source_dijkstra(G, source=0)
    print('Single source Dijkstra path and lenght: {} {}'.format(path, length))
    length, path = nx.bidirectional_dijkstra(G, source=0, target=3)
    print('Bidirectional Dijkstra path and lenght: {} {}'.format(path, length))
    '''Compute shortest path length and predecessors on shortest paths in weighted graphs.'''
    G1 = nx.cycle_graph(4)
    G1.add_edge(0, 3, weight=1)
    G1.add_edge(1, 2, weight=5)
    print('Bellman-Ford')
    draw_graph(G1)
    pred, distance = nx.bellman_ford(G, source=0, weight='weight')
    print('Bellman-Ford predecessors {} and distances {}'.format(pred, distance))
    '''Negative edge cycle'''
    print('Negative edge cycle')
    G = nx.cycle_graph(5, create_using=nx.DiGraph())
    draw_graph(G)
    print(nx.negative_edge_cycle(G))
    G[1][2]['weight'] = -7
    print(nx.negative_edge_cycle(G))
    print('Shortest path using Johnson\'s algorithm')
    graph = nx.DiGraph()
    graph.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
    paths = nx.johnson(graph, weight='weight')
    print('Shortest path using Johnson\'s algorithm: {}'.format(paths))


def shortest_path_dense_graphs_alg():
    '''a dense graph is a graph in which the number of edges is close to the maximal number of edges. The opposite,
    a graph with only a few edges, is a sparse graph.'''
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    G.add_weighted_edges_from([
        (1, 2, 7),
        (2, 4, 15),
        (4, 5, 6),
        (5, 6, 9),
        (6, 1, 14),
        (1, 3, 9),
        (2, 3, 10),
        (3, 4, 11),
        (3, 6, 2),
    ], weight='distance')
    draw_graph(G)
    paths = nx.floyd_warshall(G, weight='distance')
    print('Floyd-Warshall alg: {}'.format(paths))
    predec, distance = nx.floyd_warshall_predecessor_and_distance(G, weight='distance')
    print('Floyd-Warshall alg: {} {}'.format(predec, distance))


def shortest_path_astar_alg():
    print('Astar algorithm')
    G = nx.path_graph(5)
    print(nx.astar_path(G, 0, 4))
    draw_graph(G)
    G = nx.grid_graph(dim=[3, 3])  # nodes are two-tuples (x,y)

    def dist(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    print(nx.astar_path(G, (0, 0), (2, 2), dist))
    print(nx.astar_path_length(G, (0, 0), (2, 2), dist))
    draw_graph(G)


def simple_path_alg():
    print('Simple path')
    G = nx.complete_graph(4)
    for path in nx.all_simple_paths(G, source=0, target=3):
        print(path)
    paths = nx.all_simple_paths(G, source=0, target=3, cutoff=2)
    print(list(paths))
    draw_graph(G)
    print('Shortest simple path')
    G = nx.cycle_graph(7)
    paths = list(nx.shortest_simple_paths(G, 0, 3))
    print(paths)
    from itertools import islice
    def k_shortest_paths(G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    for path in k_shortest_paths(G, 0, 3, 2):
        print(path)
    draw_graph(G)


def swap_alg():
    print('A double-edge swap removes two randomly chosen edges u-v and x-y and creates the new edges u-x and v-y:')
    G = nx.complete_graph(4)
    draw_graph(G)
    SG = nx.double_edge_swap(G, nswap=1)
    draw_graph(SG)


def traversal_dfs_alg():
    G = nx.Graph()
    G.add_path([0, 1, 2])
    draw_graph(G)
    print('Produce edges in a depth-first-search (DFS).')
    print(list(nx.dfs_edges(G, 0)))
    print('Return oriented tree constructed from a depth-first-search from source.')
    T = nx.dfs_tree(G, 0)
    print(T.edges())
    draw_graph(T)
    print('Return dictionary of predecessors in depth-first-search from source.')
    print(nx.dfs_predecessors(G, 0))
    print('Return dictionary of successors in depth-first-search from source.')
    print(nx.dfs_successors(G, 0))
    print('Produce nodes in a depth-first-search pre-ordering starting from source')
    print(list(nx.dfs_preorder_nodes(G, 0)))
    print('Produce nodes in a depth-first-search post-ordering starting from source.')
    print(list(nx.dfs_postorder_nodes(G, 0)))
    print('Produce edges in a depth-first-search (DFS) labeled by type.')
    edges = (list(nx.dfs_labeled_edges(G, 0)))
    print(edges)


def traversal_bfs_alg():
    G = nx.Graph()
    G.add_path([0, 1, 2])
    draw_graph(G)
    print('Produce edges in a breadth-first-search starting at source.')
    print(list(nx.bfs_edges(G, 0)))
    print('Return an oriented tree constructed from of a breadth-first-search starting at source')
    T = nx.bfs_tree(G, source=0)
    print(list(nx.bfs_edges(T, 0)))
    draw_graph(T)
    print('Return dictionary of predecessors in breadth-first-search from source.')
    print(nx.bfs_predecessors(G, 0))


def traversal_edge_dfs_alg():
    print('A directed, depth-first traversal of edges in G, beginning at source.')
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]
    G = nx.Graph(edges)
    DG = nx.DiGraph(edges)
    MDG = nx.MultiDiGraph(edges)
    print(list(nx.edge_dfs(G, nodes)))
    draw_graph(G)
    print(list(nx.edge_dfs(DG, nodes)))
    print(list(nx.edge_dfs(DG, nodes, orientation='ignore')))
    draw_graph(DG)
    print(list(nx.edge_dfs(MDG, nodes)))
    print(list(nx.edge_dfs(MDG, nodes, orientation='ignore')))
    draw_graph(MDG)


def tree_alg():
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 0), (1, 0), (2, 1), (3, 1)]
    G = nx.Graph(edges)
    DG = nx.DiGraph(edges)
    print('A tree is a connected graph with no undirected cycles.')
    print('Is tree? {} '.format(nx.is_tree(G)))
    print('A forest is a graph with no undirected cycles.')
    print('Is forest? {} '.format(nx.is_forest(G)))
    print('An arborescence is a directed tree with maximum in-degree equal to 1')
    print('Is arborescence? {} '.format(nx.is_arborescence(DG)))
    print('A branching is a directed forest with maximum in-degree equal to 1')
    print('Is branching? {} '.format(nx.is_branching(DG)))
    draw_graph(G)


def tree_branching_spanning_alg():
    print('Algorithms for finding optimum branchings and spanning arborescences.')
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    G.add_weighted_edges_from([
        (1, 2, 7),
        (2, 4, 15),
        (4, 5, 6),
        (5, 6, 9),
        (6, 1, 14),
        (1, 3, 9),
        (2, 3, 10),
        (3, 4, 11),
        (3, 6, 2),
    ], weight='distance')
    DG = nx.DiGraph(G)
    draw_graph(G)
    draw_graph(DG)
    print('Branching weight: {} '.format(branching_weight(G)))
    print('Maximum branching')
    draw_graph(maximum_branching(G))
    print('Minimum branching')
    draw_graph(minimum_branching(DG))
    print('Maximum spanning arborescence')
    draw_graph(maximum_spanning_arborescence(DG))
    print('Minimum spanning arborescence')
    draw_graph(minimum_spanning_arborescence(DG))


def triads_alg():
    print('Triad - subgraph formed by tree nodes')
    print(
        'The triadic census is a count of how many of the 16 possible types of triads are present in a directed graph.')
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6])
    G.add_weighted_edges_from([
        (1, 2, 7),
        (2, 4, 15),
        (4, 5, 6),
        (5, 6, 9),
        (6, 1, 14),
        (1, 3, 9),
        (2, 3, 10),
        (3, 4, 11),
        (3, 6, 2),
    ], weight='distance')
    DG = nx.DiGraph(G)
    draw_graph(G)
    draw_graph(DG)
    print('Triad - subgraph formed by tree nodes')
    print(
        'The triadic census is a count of how many of the 16 possible types of triads are present in a directed graph.')
    print('Triadic census count: {}'.format(triadic_census(DG)))


def vitality_alg():
    print(
        'Closeness vitality of a node is the change in the sum of distances between all node pairs when excluding that node.')
    G = nx.cycle_graph(3)
    G1 = nx.Graph()
    G1.add_nodes_from([1, 2, 3, 4, 5, 6])
    G1.add_weighted_edges_from([
        (1, 2, 7),
        (2, 4, 15),
        (4, 5, 6),
        (5, 6, 9),
        (6, 1, 14),
        (1, 3, 9),
        (2, 3, 10),
        (3, 4, 11),
        (3, 6, 2),
    ], weight='distance')
    DG = nx.DiGraph(G1)
    print('Closeness vitality {}'.format(nx.closeness_vitality(G)))
    draw_graph(G)
    print('Closeness vitality {}'.format(nx.closeness_vitality(DG)))
    draw_graph(DG)

    # dijkstra_ex1()
    # repr_graph()
    # repr_graph_1()
    # repr_graph_2()
# repr_graph_3()
# create_ordered_node_graph()
# connectivity_alg()
# clique_alg()
# clique_alg1()
# average_clustering_alg()
# dominating_set_alg()
# independent_set_alg()
# matching_alg()
# ramsey_alg()
# assortativity_alg()
# bipartite_alg()
# bipartite_alg1()
# bipartite_alg2()
# bipartite_projections()
# bipartite_spectral_bipartivity()
# bipartite_clustering()
# bipartite_node_redundancy()
# bipartite_centrality()
# bipartite_graph_generators()
# blockmodel_alg()
# boundary_alg()
# centrality_alg()
# chordal_alg()
# cliques_alg2()
# clustering_alg()
# communities_alg()
# components_connectivity_alg()
# components_strong_connectivity_alg()
# components_weak_connectivity_alg()
# component_attracting_alg()
# component_biconnected_alg
# component_semiconnectedness_alg()
# connectivity_k_node_components_alg()
# connectivity_node_cuts()
# connectivity_flow_based_alg()
# core_alg()
# cycles_alg()
# directed_acyclic_graphs_alg()
# distance_measures_alg()
# distance_regular_graphs_alg()
# dominators_alg()
# eulerian_alg()
# flow_alg()
# graphical_degree_seq_alg()
# hierarchy_alg()
# hybrid_alg()
# isolates_alg()
# isomorfism_alg()
# pagerank_alg()
# hits_alg()
# link_prediction_alg()
# matching_alg()
# minors_alg()
# maximal_independent_set_alg()
# minimum_spanning_tree_alg()
# operators_alg()
# rich_club_coefficient_alg()
# shortest_path_alg()
# shortest_path_advanced_alg()
# shortest_path_dense_graphs_alg()
# shortest_path_astar_alg()
# simple_path_alg()
# swap_alg()
# traversal_dfs_alg()
# traversal_bfs_alg()
# traversal_edge_dfs_alg()
# tree_alg()
# tree_branching_spanning_alg()
# triads_alg()
vitality_alg()
