from dolfinx import fem
from ufl import TrialFunction, TestFunction, Measure
from networks_fenicsx.mesh import mesh_generation
from networks_fenicsx.config import Config

import numpy as np

n = 2

cfg = Config()
cfg.outdir = "test_diag_blocks"
cfg.export = True
cfg.lcar = 2.0

G = mesh_generation.make_tree(n=n, H=n, W=n, cfg=cfg)

submeshes = G.submeshes()

# Flux spaces on each segment, ordered by the edge list
P3s = [fem.FunctionSpace(submsh, ("Lagrange", 3)) for submsh in submeshes]
# Pressure space on global mesh
P2 = fem.FunctionSpace(G.msh, ("Lagrange", 2))

# Fluxes on each branch
qs = []
vs = []
for P3 in P3s:
    qs.append(TrialFunction(P3))
    vs.append(TestFunction(P3))
# Pressure
p = TrialFunction(P2)
phi = TestFunction(P2)

# Initialize forms
a = [[None] * (len(submeshes)) for i in range(len(submeshes))]

for i, e in enumerate(G.edges):

    submsh = G.edges[e]['submesh']
    dx_edge = Measure("dx", domain=submsh)

    a[i][i] = fem.form(qs[i] * vs[i] * dx_edge)

_A = fem.petsc.assemble_matrix_block(a)
_A.assemble()

for i in range(0, G.num_edges):
    for ii in range(i, i + 4):
        row = _A.getRow(ii)[1]
        for j in range(0, G.num_edges):
            if np.any(row[4 * j:4 * (j + 1)]):
                print("A block[", i, "] = ", row[4 * j:4 * (j + 1)])
    print("\n")
