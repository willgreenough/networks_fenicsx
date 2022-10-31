from . import mesh
from networks_fenicsx import config


def make_line_graph(n, cfg: config.Config):
    '''
    Make a graph along the unit x-axis with n nodes
    '''

    G = mesh.NetworkGraph(cfg)

    dx = 1 / (n - 1)
    print("Adding nodes 0 to ", n)
    G.add_nodes_from(range(0, n))
    for i in range(0, n):
        G.nodes[i]['pos'] = [i * dx, 0, 0]

    for i in range(0, n - 1):
        print("Adding edge [", i, ", ", i + 1, "]")
        G.add_edge(i, i + 1)

    G.build_mesh()
    G.build_network_submeshes()
    G.build_markers()

    return G


def make_Y_bifurcation(cfg: config.Config):
    '''
    Make a 3 branches network with one bifurcation
    '''

    G = mesh.NetworkGraph(cfg)

    G.add_nodes_from([0, 1, 2, 3])
    G.nodes[0]['pos'] = [0, 0, 0]
    G.nodes[1]['pos'] = [0, 0.5, 0]
    G.nodes[2]['pos'] = [-0.5, 1, 0]
    G.nodes[3]['pos'] = [0.5, 1, 0]

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.build_mesh()
    G.build_network_submeshes()
    G.build_markers()

    return G


def make_double_Y_bifurcation(cfg: config.Config):

    G = mesh.NetworkGraph(cfg)

    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    G.nodes[0]['pos'] = [0, 0, 0]
    G.nodes[1]['pos'] = [0, 0.5, 0]
    G.nodes[2]['pos'] = [-0.5, 1, 0]
    G.nodes[3]['pos'] = [0.5, 1, 0]

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.nodes[4]['pos'] = [-0.75, 1.5, 0]
    G.nodes[5]['pos'] = [-0.25, 1.5, 0]

    G.nodes[6]['pos'] = [0.25, 1.5, 0]
    G.nodes[7]['pos'] = [0.75, 1.5, 0]

    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(3, 6)
    G.add_edge(3, 7)

    G.build_mesh()
    G.build_network_submeshes()
    G.build_markers()

    return G


if __name__ == '__main__':
    make_Y_bifurcation(cfg=config.Config())
