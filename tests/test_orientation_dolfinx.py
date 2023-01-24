import dolfinx
from mpi4py import MPI
import ufl
import numpy as np

# Read pre-generated Y-shaped mesh
path_mesh = "test_orientation/mesh.xdmf"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, path_mesh, "r") as xdmf:
    mesh = xdmf.read_mesh()
    mf = xdmf.read_meshtags(mesh, "Cell tags")

submeshes = []
entity_maps = []
unique_domain = np.unique(mf.values)
for i, domain in enumerate(unique_domain):
    edge_subdomain = mf.find(domain)
    submesh, entity_map = dolfinx.mesh.create_submesh(mesh, mesh.topology.dim, edge_subdomain)[0:2]
    submeshes.append(submesh)
    entity_maps.append(entity_map)

P1s = [dolfinx.fem.FunctionSpace(submsh, ("CG", 1)) for submsh in submeshes]
P1_global = dolfinx.fem.FunctionSpace(mesh, ("CG", 1))

# Testing non-mixed integral on the RHS
print("--- TEST 1 : Non-mixed RHS ---")

for P1 in P1s:
    vs_i = ufl.TestFunction(P1)
    dx_edge = ufl.Measure("dx", domain=P1.mesh)
    x = ufl.SpatialCoordinate(P1.mesh)

    # Checking the dofmap associated with geometry of the submeshes
    geom = P1.mesh.geometry
    print("Geometry - DofMap = ", geom.dofmap.links(0))
    print("Geometry - DofMap (coords) = ", geom.x[geom.dofmap.links(0)])

    # Checking the dofmap associated with topology of the submeshes
    topo = P1.mesh.topology
    connectivity = topo.connectivity(1, 0)
    print("Topology - connectivity = ", connectivity.links(0))

    # TestFunction vs_i*(7+y), to make sure we get different values
    # - Do we get same between FEniCS and FEniCSx ? + Find out the possible permutations (dofs)
    L_i = dolfinx.fem.form(vs_i * (7 + x[1]) * dx_edge)
    print("RHS[vs_i*(7+x[1])] = ", dolfinx.fem.petsc.assemble_vector(L_i).array)

    # Derivative of TestFunction
    # - Could there be an issue with the derivatives / grap operators (Jacobian)
    L_i_dy = dolfinx.fem.form(vs_i.dx(1) * dx_edge)
    print("RHS[vs_i.dx(1)] = ", dolfinx.fem.petsc.assemble_vector(L_i_dy).array)

# Checking the dofmap associated with geometry of the parent (global) mesh
geom = P1_global.mesh.geometry
for c in range(geom.dofmap.num_nodes):
    print("Geometry - DofMap = ", geom.dofmap.links(c))
    print("Geometry - DofMap (coords) = ", geom.x[geom.dofmap.links(c)])

print("------------------------------ \n \n")

# Testing non-diagonal block (mixed) on a matrix
print("--- TEST 2 : Mixed matrix / Off-diag block ---")

p = ufl.TrialFunction(P1_global)
phi = ufl.TestFunction(P1_global)

for P1, imap in zip(P1s, entity_maps):
    entity_maps = {P1_global.mesh: imap}
    qs_i = ufl.TrialFunction(P1)
    vs_i = ufl.TestFunction(P1)
    dx_edge = ufl.Measure("dx", domain=P1.mesh)
    x = ufl.SpatialCoordinate(P1.mesh)

    # TestFunction * TrialFunction => Block[0,1] : vs_i*p
    a_i = [[dolfinx.fem.form(dolfinx.fem.Constant(P1.mesh, 0.0) * qs_i * vs_i * dx_edge),
            dolfinx.fem.form(vs_i * p * dx_edge, entity_maps=entity_maps)],
           [dolfinx.fem.form(dolfinx.fem.Constant(P1.mesh, 0.0) * qs_i * phi * dx_edge, entity_maps=entity_maps),
            dolfinx.fem.form(dolfinx.fem.Constant(P1_global.mesh, 0.0) * p * phi * ufl.dx)]]

    A_i = dolfinx.fem.petsc.assemble_matrix_block(a_i)
    A_i.assemble()
    print("A[vs_i*p] = ", A_i.view())

    # Derivative of TestFunction * TrialFunction => Block[0,1] : vs_i.dx(1)*p
    a_i_dy = [[dolfinx.fem.form(dolfinx.fem.Constant(P1.mesh, 0.0) * qs_i * vs_i * dx_edge),
               dolfinx.fem.form(vs_i.dx(1) * p * dx_edge, entity_maps=entity_maps)],
              [dolfinx.fem.form(dolfinx.fem.Constant(P1.mesh, 0.0) * qs_i * phi * dx_edge, entity_maps=entity_maps),
               dolfinx.fem.form(dolfinx.fem.Constant(P1_global.mesh, 0.0) * p * phi * ufl.dx)]]

    A_i_dy = dolfinx.fem.petsc.assemble_matrix_block(a_i_dy)
    A_i_dy.assemble()
    print("A[vs_i.dx(1)*p] = ", A_i_dy.view())

    # Checking the Jacobian
    jacobian_P1 = ufl.Jacobian(P1.mesh)
    M_P1 = ufl.as_matrix([[1 + 3 * i + j for i in range(1)] for j in range(3)])
    Jac_dot_M = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(jacobian_P1, M_P1) * dx_edge))
    print("dot(Jacobian(P1), M) = ", Jac_dot_M)

print("----------------------------------------------")
