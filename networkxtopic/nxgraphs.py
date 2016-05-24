import matplotlib.pyplot as plt
import networkx as nx


def draw_graph(g):
    nx.draw(g, nodecolor='r', edge_color='b', with_labels=True, data='weight')
    plt.savefig("repr_graph.png")
    plt.show()


'''
def atlas_graph_types():
    G = graph_atlas_g()
    draw_graph(G)
'''


def classic_graphs():
    print("Balanced Tree")
    BG = nx.balanced_tree(3, 2)
    draw_graph(BG)
    print("Barbell Graph")
    BBG = nx.barbell_graph(3, 2)
    draw_graph(BBG)
    print("Complete Graph")
    CG = nx.complete_graph(10)
    draw_graph(CG)
    print("Complete Multipartite Graph")
    CMG = nx.complete_multipartite_graph(1, 2, 10)
    print([CMG.node[u]['block'] for u in CMG])
    print(CMG.edges(0))
    print(CMG.edges(2))
    print(CMG.edges(4))
    draw_graph(CMG)
    print("Circular Ladder Graph")
    CLG = nx.circular_ladder_graph(5)
    draw_graph(CLG)
    print("Dorogovtsev Goltsev Mendes Graph")
    DGMG = nx.dorogovtsev_goltsev_mendes_graph(3)
    draw_graph(DGMG)
    print("Empty Graph")
    EG = nx.empty_graph(5, create_using=nx.DiGraph())
    draw_graph(EG)
    print("Grid 2D Graph")
    G2DG = nx.grid_2d_graph(5, 6)
    draw_graph(G2DG)
    print("Grid Graph")
    GDG = nx.grid_graph(dim=[5, 2])
    draw_graph(GDG)
    print("Hypercube Graph")
    HG = nx.hypercube_graph(3)
    draw_graph(HG)
    print("Ladder Graph")
    LG = nx.ladder_graph(8)
    draw_graph(LG)
    print("Ladder Graph")
    LG = nx.ladder_graph(8)
    draw_graph(LG)
    print("Lollipop Graph")
    LPG = nx.lollipop_graph(n=6, m=4)
    draw_graph(LPG)
    print("Null Graph")
    NG = nx.null_graph()
    draw_graph(NG)
    print("Path Graph")
    PG = nx.path_graph(16)
    draw_graph(PG)
    print("Star Graph")
    SG = nx.star_graph(16)
    draw_graph(SG)
    print("Trivial Graph")
    TG = nx.trivial_graph()
    draw_graph(TG)
    print("Wheel Graph")
    WG = nx.wheel_graph(n=18)
    draw_graph(WG)


def expanders_graphs():
    print("Margulis - Gabber - Galil Graph")
    MGGG = nx.margulis_gabber_galil_graph(n=3)
    draw_graph(MGGG)
    print("Chordal - Cycle Graph")
    MGGG = nx.chordal_cycle_graph(8)
    draw_graph(MGGG)


def small_graphs():
    ''' '''


# atlas_graph_types()
# classic_graphs()
# expanders_graphs()
small_graphs()
