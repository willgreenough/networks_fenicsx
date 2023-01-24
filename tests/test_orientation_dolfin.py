# flake8: noqa
# type: ignore

from dolfin import *
import ufl
import numpy as np

nb_edges = 3
mesh = Mesh()
mesh_file = "test_orientation/mesh.xdmf"

# Read pre-generated Y-shaped mesh
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

P1s = [FunctionSpace(msh, 'CG', 1) for msh in submeshes]
P1_global = FunctionSpace(mesh, 'CG', 1)

# Testing non-mixed integral on the RHS
print("--- TEST 1 : Non-mixed RHS ---")

for P1 in P1s:
    vs_i = TestFunction(P1)
    dx_edge = Measure('dx', domain=P1.mesh())
    x = SpatialCoordinate(P1.mesh())

    # Checking the dofmap associated with geometry of the submeshes
    cells = P1.mesh().cells()
    coords = P1.mesh().coordinates()[cells[0]]
    print("Geometry - DofMap = ", cells)
    print("Geometry - DofMap (coords) = ", coords)

    # Checking the dofmap associated with topology of the submeshes
    P1.mesh().init(1,0)
    connectivity = P1.mesh().topology()(1,0)
    print("Topology - connectivity = ", connectivity(0))

    # TestFunction vs_i*(7+y), to make sure we get different values
    # - Do we get same between FEniCS and FEniCSx ? + Find out the possible permutations (dofs)
    L_i = vs_i*(7+x[1])*dx_edge
    print("RHS[vs_i*(7+x[1])] = ", assemble(L_i).get_local())

    # Derivative of TestFunction
    # - Could there be an issue with the derivatives / grap operators (Jacobian)
    L_i_dy = vs_i.dx(1)*dx_edge
    print("RHS[vs_i.dx(1)] = ", assemble(L_i_dy).get_local())

# Checking the dofmap associated with geometry of the parent (global) mesh 
cells = P1_global.mesh().cells()
for c in cells:
    coords = P1_global.mesh().coordinates()[c]
    print("Geometry - DofMap = ", c)
    print("Geometry - DofMap (coords) = ", coords)

print("------------------------------ \n \n")

# Testing non-diagonal block (mixed) on a matrix
print("--- TEST 2 : Mixed matrix / Off-diag block ---")

for P1 in P1s:
    # Function space
    W = MixedFunctionSpace(P1, P1_global)

    # Trial and test functions
    (vs_i, phi) = TestFunctions(W)
    (qs_i, p) = TrialFunctions(W)
    qp0 = Function(W)
    dx_edge = Measure('dx', domain=P1.mesh())
    x = SpatialCoordinate(P1.mesh())

    a_i_base = Constant(0.)*qs_i*vs_i*dx_edge
    a_i_base += Constant(0)*p*phi*dx_edge
    L_i = Constant(0.)*vs_i*dx_edge

    # TestFunction * TrialFunction => Block[0,1] : vs_i*p
    a_i = a_i_base + vs_i*p*dx_edge

    system = assemble_mixed_system(a_i == L_i, qp0)
    A_list = system[0]
    print("A[vs_i*p] = ", A_list[1].str(True))

    # Derivative of TestFunction * TrialFunction => Block[0,1] : vs_i.dx(1)*p
    a_i_dy = a_i_base + vs_i.dx(1)*p*dx_edge

    system_dy = assemble_mixed_system(a_i_dy == L_i, qp0)
    A_list_dy = system_dy[0]
    print("A[vs_i.dx(1)*p] = ", A_list_dy[1].str(True))

    # Checking the Jacobian
    jacobian_P1 = ufl.Jacobian(P1.mesh())
    M_P1 = ufl.as_matrix([ [1+3*i+j for i in range(1)] for j in range(3)])
    Jac_dot_M = assemble( ufl.inner(jacobian_P1, M_P1)*dx_edge )
    print("dot(Jacobian(P1), M) = ", Jac_dot_M)

print("----------------------------------------------")
