from dolfinx import fem, io
from petsc4py import PETSc

from networks_fenicsx.mesh import mesh
from networks_fenicsx.solver import assembly
from networks_fenicsx.utils.timers import timeit
from networks_fenicsx.utils.mesh_utils import transfer_submesh_data
from networks_fenicsx import config


class Solver():

    def __init__(self, config: config.Config, graph: mesh.NetworkGraph, assembler: assembly.Assembler):
        self.G = graph
        self.assembler = assembler
        self.cfg = config

        if self.assembler is not None:
            self.A = assembler.assembled_matrix()
            self.b = assembler.assembled_rhs()

    @timeit
    def solve(self):

        # FIXME : To be given as options
        # Configure solver
        ksp = PETSc.KSP().create(self.G.msh.comm)
        ksp.setOperators(self.A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("superlu_dist")

        # Solve
        x = self.A.createVecLeft()
        ksp.solve(self.b, x)

        q_degree = self.assembler.function_spaces[0].element.basix_element.degree
        global_q_space = fem.FunctionSpace(self.G.msh, ("DG", q_degree))
        global_q = fem.Function(global_q_space)

        # Recover solution
        fluxes = []
        start = 0
        for i, e in enumerate(self.G.edges):
            q_space = self.assembler.function_spaces[i]
            q = fem.Function(q_space)
            offset = q_space.dofmap.index_map.size_local * q_space.dofmap.index_map_bs
            q.x.array[:offset] = x.array_r[start:start + offset]
            q.x.scatter_forward()
            start += offset
            fluxes.append(q)

            # Interpolated to DG space
            DG_q_space = fem.FunctionSpace(q_space.mesh, ("DG", q_degree))
            DG_q = fem.Function(DG_q_space)
            DG_q.interpolate(q)
            # Transferring from DG submesh to DG parent
            transfer_submesh_data(global_q, DG_q, self.G.edges[e]['entity_map'], inverse=True)

        print("global flux = ", global_q.x.array)

        p_space = self.assembler.function_spaces[-1]
        offset = p_space.dofmap.index_map.size_local * p_space.dofmap.index_map_bs
        pressure = fem.Function(p_space)
        pressure.x.array[:(len(x.array_r) - start)] = x.array_r[start:start + offset]
        pressure.x.scatter_forward()
        print("pressure = ", pressure.x.array)

        for i, q in enumerate(fluxes):
            with io.XDMFFile(self.G.msh.comm, self.cfg.outdir + "/results/flux_" + str(i) + ".xdmf", "w") as file:
                file.write_mesh(q.function_space.mesh)
                file.write_function(q)

        with io.XDMFFile(self.G.msh.comm, self.cfg.outdir + "/results/flux.xdmf", "w") as file:
            file.write_mesh(global_q.function_space.mesh)
            file.write_function(global_q)

        with io.XDMFFile(self.G.msh.comm, self.cfg.outdir + "/results/pressure.xdmf", "w") as file:
            file.write_mesh(pressure.function_space.mesh)
            file.write_function(pressure)

        return (fluxes, pressure)
