from dolfinx import fem
from petsc4py import PETSc

from networks_fenicsx.mesh import mesh
from networks_fenicsx.solver import assembly


class Solver():

    def __init__(self, graph: mesh.NetworkGraph, assembler: assembly.Assembler):
        self.G = graph
        self.assembler = assembler

        if self.assembler is not None:
            self.A = assembler.assembled_matrix()
            self.b = assembler.assembled_rhs()

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
            print("q[", i, "] = ", q.x.array)
            fluxes.append(q)

        p_space = self.assembler.function_spaces[-1]
        offset = p_space.dofmap.index_map.size_local * p_space.dofmap.index_map_bs
        pressure = fem.Function(p_space)
        pressure.x.array[:(len(x.array_r) - start)] = x.array_r[start:start + offset]
        pressure.x.scatter_forward()
        print("pressure = ", pressure.x.array)

        return (fluxes, pressure)
