# flake8: noqa
# type: ignore
from dolfin import *

nb_edges = 3

mesh = Mesh()
mesh_file = "test_diag_blocks/mesh/mesh.xdmf"  # Mesh generated from test_diag_blocks_fenicsx

# Read mesh generated fron FEniCSx
with XDMFFile(MPI.comm_world, mesh_file) as xdmf:
    xdmf.read(mesh)

# Make meshfunction containing edge ixs
mf = MeshFunction('size_t', mesh, 1)
mf.array()[:] = range(0, nb_edges)

submeshes = []
# Make and store one submesh for each edge
for i in range(0, nb_edges):
    submsh = MeshView.create(mf, i)
    submeshes.append(submsh)

P3s = [FunctionSpace(msh, 'CG', 3) for msh in submeshes]
P2 = FunctionSpace(mesh, 'CG', 2)  # Pressure space (on whole mesh)

# Function spaces
spaces = P3s + [P2]
W = MixedFunctionSpace(*spaces)

# Trial and test functions
vphi = TestFunctions(W)
qp = TrialFunctions(W)

# split out the components
qs = qp[0:nb_edges]
p = qp[-1]

vs = vphi[0:nb_edges]
phi = vphi[-1]

# Initialize blocks in a and L to zero
# (so fenics-mixed-dim does not throw an error)
dx = Measure('dx', domain=mesh)
a = Constant(0) * p * phi * dx
L = Constant(0) * phi * dx

# Assemble edge contributions to a and L
for i in range(0, nb_edges):
    submsh = submeshes[i]
    dx_edge = Measure("dx", domain=submsh)

    a += qs[i] * vs[i] * dx_edge

# Assemble the system
qp0 = Function(W)
system = assemble_mixed_system(a == L, qp0)

A_list = system[0]
rhs_blocks = system[1]

_A = PETScNestMatrix(A_list)  # recombine blocks
_A.convert_to_aij()  # Convert MATNEST to AIJ
for i in range(0, nb_edges):
    for ii in range(i, i + 4):
        row = _A.mat().getRow(ii)[1][0:4]
        print("A block [", i, "]= ", row)
    print("\n")
