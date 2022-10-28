from petsc4py import PETSc
import scipy.sparse as sp


def convert_vec_to_petscmatrix(vec):
    '''
    Convert a fenics vector from assemble into PETSc.Mat()
    '''

    # Make sparse
    sparse_vec = sp.coo_matrix(vec)

    # Init PETSC matrix
    petsc_mat = PETSc.Mat().createAIJ(size=sparse_vec.shape)
    petsc_mat.setUp()

    # Input values from sparse_vec
    for i, j, v in zip(sparse_vec.row, sparse_vec.col, sparse_vec.data):
        petsc_mat.setValue(i, j, v)

    petsc_mat.assemble()
    return petsc_mat
