from petsc4py import PETSc
import scipy.sparse as sp

'''
This file is based on the graphnics project (https://arxiv.org/abs/2212.02916), https://github.com/IngeborgGjerde/fenics-networks - forked on August 2022
Copyright (C) 2022-2023 by Ingeborg Gjerde

You can freely redistribute it and/or modify it under the terms of the GNU General Public License, version 3.0, provided that the above copyright notice is kept intact and that the source code is made available under an open-source license.
'''

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
