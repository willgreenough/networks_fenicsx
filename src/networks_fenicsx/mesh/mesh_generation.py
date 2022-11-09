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


def tree_edges(n, r):
    # helper function for trees
    # yields edges in rooted tree at 0 with n nodes and branching ratio r
    if n == 0:
        return
    # Root branch
    source = 0
    target = 1
    yield source, target
    # Other branches
    nodes = iter(range(1, n))
    parents = [next(nodes)]  # stack of max length r
    while parents:
        source = parents.pop(0)
        for i in range(r):
            try:
                target = next(nodes)
                parents.append(target)
                yield source, target
            except StopIteration:
                break


def make_tree(n: int, H: float, W: float, cfg: config.Config):
    '''
    n : number of generations
    H : height
    W : width
    '''

    # FIXME : add parameter r : branching factor of the tree (each node has r children)
    r = 2
    G = mesh.NetworkGraph(cfg)

    nb_nodes_gen = []
    for i in range(n):
        nb_nodes_gen.append(pow(r, i))

    nb_nodes = 1 + sum(nb_nodes_gen)
    nb_nodes_last = pow(r, n - 1)

    G.add_nodes_from(range(nb_nodes))

    x_offset = W / (2 * (nb_nodes_last - 1))
    y_offset = H / n

    # Add two first nodes
    idx = 0
    G.nodes[idx]['pos'] = [0, 0, 0]
    G.nodes[idx + 1]['pos'] = [0, y_offset, 0]
    idx = idx + 2

    # Add nodes for rest of the tree
    for gen in range(1, n):
        factor = pow(2, n - gen)
        x = x_offset * (factor / 2)
        y = y_offset * (gen + 1)
        x_coord = []
        nb_nodes_ = int(nb_nodes_gen[gen] / 2)
        for i in range(nb_nodes_):
            x_coord.append(x)
            x_coord.append(-x)
            x = x + x_offset * factor
        # Add nodes to G, from sorted x_coord array
        x_coord.sort()
        for x in x_coord:
            G.nodes[idx]['pos'] = [x, y, 0]
            idx = idx + 1

    edges = tree_edges(nb_nodes, r)
    for (e0, e1) in list(edges):
        G.add_edge(e0, e1)

    G.build_mesh()
    G.build_network_submeshes()
    G.build_markers()

    return G


if __name__ == '__main__':
    make_Y_bifurcation(cfg=config.Config())
