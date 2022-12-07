from dolfinx import fem, io
from mpi4py import MPI

from networks_fenicsx.mesh import mesh
from networks_fenicsx import config
from networks_fenicsx.utils.mesh_utils import transfer_submesh_data


def export(config: config.Config, graph: mesh.NetworkGraph, function_spaces: list, sol):

    q_space = function_spaces[0]
    p_space = function_spaces[-1]
    q_degree = q_space.element.basix_element.degree
    global_q_space = fem.FunctionSpace(graph.msh, ("DG", q_degree))
    global_q = fem.Function(global_q_space)

    # Recover solution
    fluxes = []
    start = 0
    for i, e in enumerate(graph.edges):
        q_space = function_spaces[i]
        q = fem.Function(q_space)
        offset = q_space.dofmap.index_map.size_local * q_space.dofmap.index_map_bs
        q.x.array[:offset] = sol.array_r[start:start + offset]
        q.x.scatter_forward()
        start += offset
        fluxes.append(q)

        # Interpolated to DG space
        DG_q_space = fem.FunctionSpace(q_space.mesh, ("DG", q_degree))
        DG_q = fem.Function(DG_q_space)
        DG_q.interpolate(q)
        # Transferring from DG submesh to DG parent
        transfer_submesh_data(global_q, DG_q, graph.edges[e]['entity_map'], inverse=True)

    offset = p_space.dofmap.index_map.size_local * p_space.dofmap.index_map_bs
    pressure = fem.Function(p_space)
    pressure.x.array[:(len(sol.array_r) - start)] = sol.array_r[start:start + offset]
    pressure.x.scatter_forward()

    for i, q in enumerate(fluxes):
        with io.XDMFFile(MPI.COMM_WORLD, config.outdir + "/results/flux_" + str(i) + ".xdmf", "w") as file:
            file.write_mesh(q.function_space.mesh)
            file.write_function(q)

    with io.XDMFFile(MPI.COMM_WORLD, config.outdir + "/results/flux.xdmf", "w") as file:
        file.write_mesh(global_q.function_space.mesh)
        file.write_function(global_q)

    with io.XDMFFile(MPI.COMM_WORLD, config.outdir + "/results/pressure.xdmf", "w") as file:
        file.write_mesh(pressure.function_space.mesh)
        file.write_function(pressure)

    return (fluxes, global_q, pressure)
