import os
from operator import add
from ufl import TrialFunction, TestFunction, dx, inner, dot, grad, Constant, Measure
from mpi4py import MPI
from dolfinx import fem, io, mesh
from petsc4py import PETSc
import logging

from networks_fenicsx.mesh import mesh
from networks_fenicsx.utils import petsc_utils

class Assembler():

    def __init__(self, graph: mesh.NetworkGraph):
        self.G = graph
        self.function_spaces = None
        self.a = None
        self.L = None
        self.A = None
        self.L = None
        
    def dds(self, f):
        '''
        function for derivative df/ds along graph
        '''
        return dot(grad(f), self.G.global_tangent)

    def jump_vector(self, q, ix, j):
        '''
        Returns the signed jump vector for a flux function q on edge ix 
        over bifurcation j
        '''
        
        edge_list = list(self.G.edges.keys())
        
        # Iitialize form to zero
        zero = fem.Function(q.ufl_function_space())        
        L = zero*q*dx

        # Add point integrals (jump)
        for i, e in enumerate(self.G.in_edges(j)):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            edge_ix = edge_list.index(e)
            if ix==edge_ix: L += q*ds_edge(self.G.BIF_IN)

        for i, e in enumerate(self.G.out_edges(j)):
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'], subdomain_data=self.G.edges[e]['vf'])
            edge_ix = edge_list.index(e)
            if ix==edge_ix: L -= q*ds_edge(self.G.BIF_OUT)

        L = fem.form(L)
        b = fem.petsc.assemble_vector(L)

        return b

    def compute_forms(self, f=None, p_bc_ex=None):
        '''
        Solve hydraulic network model 
            R q + d/ds p = 0
            d/ds q = f
        on graph G, with bifurcation condition q_in = q_out
        and custom assembly of the bifurcatio condition 

        Args:
           f (df.function): source term
           p_bc (df.function): neumann bc for pressure
        '''
        
        if f is None:
            f = Constant(self.G.msh, 0)

        submeshes = self.G.submeshes()
            
        # Flux spaces on each segment, ordered by the edge list
        P3s = [fem.FunctionSpace(submsh, ("Lagrange", 3)) for submsh in submeshes]
        # Pressure space on global mesh
        P2 = fem.FunctionSpace(self.G.msh, ("Lagrange", 2))
        self.function_spaces = P3s + [P2]

        # Fluxes on each branch
        qs = []
        vs = []
        for P3 in P3s:
            qs.append(TrialFunction(P3))
            vs.append(TestFunction(P3))
        # Pressure
        p = TrialFunction(P2)
        phi = TestFunction(P2)

        ## Assemble variational formulation 
        dx = Measure('dx', domain=self.G.msh)

        # Compute jump vectors to be added into the global matrix as Lagrange multipliers
        self.jump_vectors = [[self.jump_vector(q, ix, j) for j in self.G.bifurcation_ixs] for ix, q in enumerate(qs)]
    
        # Initialize forms
        self.a = [[None]*(len(submeshes) + 1) for i in range(len(submeshes) + 1)]
        self.L = [None]*(len(submeshes) + 1)

        # Build the global entity map
        edge_imap = self.G.msh.topology.index_map(self.G.msh.topology.dim)
        num_edges = edge_imap.size_local + edge_imap.num_ghosts
        entity_maps = {}
        for i, e in enumerate(self.G.edges):
            submsh = self.G.edges[e]['submesh']
            entity_map = self.G.edges[e]['entity_map']
            entity_maps.update({submsh: [entity_map.index(entity)
                                         if entity in entity_map else -1
                                         for entity in range(num_edges)]})
    
        # Assemble edge contributions to a and L
        for i, e in enumerate(self.G.edges):
            dx_edge = Measure("dx", domain = self.G.edges[e]['submesh'])
            ds_edge = Measure('ds', domain=self.G.edges[e]['submesh'],
                              subdomain_data=self.G.edges[e]['vf'])


            self.a[i][i] = fem.form(qs[i]*vs[i]*dx_edge)
            self.a[-1][i] = fem.form(phi*self.dds(qs[i])*dx, entity_maps=entity_maps)
            self.a[i][-1] = fem.form(-p*self.dds(vs[i])*dx, entity_maps=entity_maps)

            # Boundary condition on the correct space
            P1_e = fem.FunctionSpace(self.G.edges[e]['submesh'], ("Lagrange", 1))
            p_bc = fem.Function(P1_e)
            p_bc.interpolate(p_bc_ex.eval)

            self.L[i] = fem.form(p_bc*vs[i]*ds_edge(self.G.BOUN_IN) - p_bc*vs[i]*ds_edge(self.G.BOUN_OUT),
                                 entity_maps=entity_maps)
        
        # Add zero to uninitialized diagonal blocks (needed by petsc)
        zero = fem.Function(P2)
        self.a[-1][-1] = fem.form(zero*p*phi*dx)
        self.L[-1] = fem.form(zero*phi*dx)

    def assemble(self):
        # Get the forms
        a = self.bilinear_forms()
        L = self.linear_forms()
        # Assemble system from the given forms
        _A = fem.petsc.assemble_matrix_block(a)
        _A.assemble()
        _b = fem.petsc.assemble_vector_block(L, a)
        
        # Get  values form A to be inserted in new bigger matrix A_new
        _A_size = _A.getSize()
        _b_size = _b.getSize()
        _A_values = _A.getValues(range(_A_size[0]), range(_A_size[1]))
        _b_values = _b.getValues(range(_b_size))

        # Build new system to include Lagrange multipliers for the bifurcation conditions
        num_bifs = len(self.G.bifurcation_ixs)
        A = PETSc.Mat().create()
        A.setSizes(list( map(add, _A_size, (num_bifs, num_bifs)) ))
        A.setUp()

        b = PETSc.Vec().create()
        b.setSizes(_b_size + num_bifs)
        b.setUp()

        # Copy _A and _b values into (bigger) system
        A.setValuesBlocked(range(_A_size[0]), range(_A_size[1]), _A_values)
        b.setValuesBlocked(range(_b_size), _b_values)

        # Convert to PETSc.Mat() object
        jump_vecs = [[petsc_utils.convert_vec_to_petscmatrix(b_row) for b_row in qi] for qi in self.jump_vectors]

        # Insert jump vectors into A_new
        for idx_block, v in enumerate(jump_vecs):
            for idx_bif, jump_vec in enumerate(v):
                # Compute jump vector transpose
                jump_vec_T = jump_vec.copy()
                jump_vec_T.transpose()
                # Add jump vector and its transpose to correct matrix block
                jump_vec_values = jump_vec.getValues(range(jump_vec.getSize()[0]), range(jump_vec.getSize()[1]))
                jump_vec_T_values = jump_vec_T.getValues(range(jump_vec_T.getSize()[0]), range(jump_vec_T.getSize()[1]))
            
                A.setValuesBlocked(idx_bif + _A_size[0],
                                   range(jump_vec.getSize()[1]*idx_block, jump_vec.getSize()[1]*(idx_block + 1)),
                                   jump_vec_values)
                A.setValuesBlocked(range(jump_vec_T.getSize()[0]*idx_block, jump_vec_T.getSize()[0]*(idx_block + 1)),
                                   idx_bif + _A_size[1],
                                   jump_vec_T_values)

        # Assembling A and b
        A.assemble()
        b.assemble()

        self.A = A
        self.b = b

        return (A,b)

    def bilinear_forms(self):
        if self.a is None:
            logging.error("Bilinear forms haven't been computed. Need to call compute_forms()")
        else:
            return self.a

    def bilinear_form(self, i:int, j:int):
        a = self.bilinear_forms()
        if i > len(a) or j > len(a[i]):
            logging.error("Bilinear form a[" + str(i) + "][" + str(j) + "] out of range")
        return a[i][j]
    
    def linear_forms(self):
        if self.L is None:
            logging.error("Linear forms haven't been computed. Need to call compute_forms()")
        else:
            return self.L

    def linear_form(self, i:int):
        L = self.linear_forms()
        if i > len(L):
            logging.error("Linear form L[" + str(i) + "] out of range")
        return L[i]

    def assembled_matrix(self):
        if self.A is None:
            logging.error("Matrix has not been assemble. Need to call assemble()")
        else:
            return self.A

    def assembled_rhs(self):
        if self.b is None:
            logging.error("RHS has not been assemble. Need to call assemble()")
        else:
            return self.b

