import os
import numpy as np

from dolfinx import io
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.solver import assembly, solver

# Clear fenics cache
print('Clearing cache')
os.system('dijitso clean') 

# Create Y bifurcation graph
G = mesh_generation.make_Y_bifurcation()

class p_bc_expr:
    def eval(self, x):
        return np.full(x.shape[1], x[0])

assembler = assembly.Assembler(G)
assembler.compute_forms(p_bc_ex=p_bc_expr())
assembler.assemble()

solver = solver.Solver(G, assembler)
(fluxes, pressure) = solver.solve()

# Write to file
for i,q in enumerate(fluxes):
    with io.XDMFFile(G.msh.comm, "demo_Y_bifurcation/flux_" + str(i) + ".xdmf", "w") as file:
        file.write_mesh(q.function_space.mesh)
        file.write_function(q)

with io.XDMFFile(G.msh.comm, "demo_Y_bifurcation/pressure.xdmf", "w") as file:
    file.write_mesh(pressure.function_space.mesh)
    file.write_function(pressure)
