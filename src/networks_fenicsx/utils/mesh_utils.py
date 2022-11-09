import numpy as np

from dolfinx import fem


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
