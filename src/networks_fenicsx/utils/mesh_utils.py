import numpy as np

from dolfinx import fem
from dolfinx.cpp.mesh import cell_num_entities

'''
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.

Modified by Cécile Daversin-Catty - 2023
'''

def reorder_mesh(msh):
    # FIXME Check this is correct
    # FIXME For a high-order mesh, the geom has more dofs so need to modify
    # this
    # FIXME What about quads / hexes?
    tdim = msh.topology.dim
    num_cell_vertices = cell_num_entities(msh.topology.cell_type, 0)
    c_to_v = msh.topology.connectivity(tdim, 0)
    geom_dofmap = msh.geometry.dofmap
    vertex_imap = msh.topology.index_map(0)
    geom_imap = msh.geometry.index_map()
    for i in range(0, len(c_to_v.array), num_cell_vertices):
        topo_perm = np.argsort(vertex_imap.local_to_global(
            c_to_v.array[i:i + num_cell_vertices]))
        geom_perm = np.argsort(geom_imap.local_to_global(
            geom_dofmap.array[i:i + num_cell_vertices]))

        c_to_v.array[i:i + num_cell_vertices] = \
            c_to_v.array[i:i + num_cell_vertices][topo_perm]
        geom_dofmap.array[i:i + num_cell_vertices] = \
            geom_dofmap.array[i:i + num_cell_vertices][geom_perm]


def transfer_submesh_data(u_parent: fem.Function, u_sub: fem.Function,
                          sub_to_parent_cells: np.ndarray, inverse: bool = False):
    """
    Transfer data between a function from the parent mesh and a function from the sub mesh.
    Both functions has to share the same element dof layout
    Args:
        u_parent: Function on parent mesh
        u_sub: Function on sub mesh
        sub_to_parent_cells: Map from sub mesh (local index) to parent mesh (local index)
        inverse: If true map from u_sub->u_parent else u_parent->u_sub
    """
    V_parent = u_parent.function_space
    V_sub = u_sub.function_space
    # FIXME: In C++ check elementlayout for equality
    if inverse:
        for i, cell in enumerate(sub_to_parent_cells):
            bs = V_parent.dofmap.bs
            bs_sub = V_sub.dofmap.bs
            assert (bs == bs_sub)
            parent_dofs = V_parent.dofmap.cell_dofs(cell)
            sub_dofs = V_sub.dofmap.cell_dofs(i)
            for p_dof, s_dof in zip(parent_dofs, sub_dofs):
                for j in range(bs):
                    u_parent.x.array[p_dof * bs + j] = u_sub.x.array[s_dof * bs + j]
    else:
        for i, cell in enumerate(sub_to_parent_cells):
            bs = V_parent.dofmap.bs
            bs_sub = V_sub.dofmap.bs
            assert (bs == bs_sub)
            parent_dofs = V_parent.dofmap.cell_dofs(cell)
            sub_dofs = V_sub.dofmap.cell_dofs(i)
            for p_dof, s_dof in zip(parent_dofs, sub_dofs):
                for j in range(bs):
                    u_sub.x.array[s_dof * bs + j] = u_parent.x.array[p_dof * bs + j]
