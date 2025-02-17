import numpy as np
from scipy.linalg import svd
from itertools import product
from numba import njit, prange

def InitialState(states_, coeffs_, dim_):
    C_ = np.zeros(dim_) #tensor C_{j1 j2 j3 j4}
    for i in range(len(states_)):
        C_[tuple(states_[i, :])] = coeffs_[i]
    return C_/np.sqrt(np.vdot(coeffs_, coeffs_))
                 

def GroupIndex(indexes_):

    labels_ = []
    newindex_ = []
    count = 0
    for label in product(range(0, 2), repeat=len(indexes_)): 
        labels_.append(list(label))
        newindex_.append(count)

        count += 1

    return np.array(labels_), np.array(newindex_)

def GroupLeftIndex(index_, bonddim_):

    labels_ = []
    newindex_ = []
    count = 0
    for i in range(len(index_)):
        for ii in range(bonddim_):
            labels_.append([i, ii])
            newindex_.append(count)

            count += 1

    return np.array(labels_), np.array(newindex_)


def Tensor2Matrix(Nsites_, indexes_, states_, coeffs_, dim_):
    C_ = InitialState(states_, coeffs_, dim_)
    labels_, newindex_ = GroupIndex(indexes_[-(Nsites_-1):, :])

    M_ = np.zeros((len(indexes_[0, :]), len(newindex_)))

    for i in range(len(indexes_[0])):
        for ii in range(len(newindex_)):
            M_[i, ii] = C_[(i,) + tuple(labels_[ii])]
    return M_

def ApplySVD(matrix_):
    return svd(matrix_, full_matrices=False) #Decomposicao em valor singular da primeira matriz

def SweepSite(Nsites_, site_, bonddim_, indexes_, matrix_):
    Right_indexes_ = indexes_[-(Nsites_-site_-1):, :]

    Rlabels_, k1_ = GroupIndex(Right_indexes_)
    Llabels_, k2_ = GroupLeftIndex(indexes_[(Nsites_-site_-1), :], bonddim_)

    M_ = np.zeros((len(k2_), len(k1_)))

    for i in range(len(k2_)):
        for ii in range(len(k1_)):

            #print(Llabels_[i, 1], len(k1_)*Llabels_[i, 0]+k1_[ii])
            M_[i, ii] = matrix_[Llabels_[i, 1], len(k1_)*Llabels_[i, 0]+k1_[ii]]

    return M_

def CalculatePreMatrices(Nsites_, indexes_, states_, coeffs_, dim_):
    M_ = Tensor2Matrix(Nsites_, indexes_, states_, coeffs_, dim_)
    U_, Sigma_, Vdag_ = ApplySVD(M_)

    Us_ = []
    SingVals_ = []
    for i in range(1, Nsites_-1):
        U_, Sigma_, Vdag_ = ApplySVD(M_)

        a_ = len(Sigma_)
        M_ = SweepSite(Nsites_, i, a_, indexes_, Vdag_)

        Us_.append(U_)
        SingVals_.append(Sigma_)

    U_, Sigma_, Vdag_ = ApplySVD(M_)

    Us_.append(U_)
    Us_.append(Vdag_)
    SingVals_.append(Sigma_)

    return Us_, SingVals_

def ContractSVals(Us_, SingVals_):
    As_ = []
    for i in range(len(Us_)-1):
        As_.append(np.array(np.split(np.matmul(Us_[i], np.diag(SingVals_[i])), 2)))
    As_.append(np.array(np.split(Us_[-1], 2, axis = 1)))

    return As_

@njit(fastmath=True)
def nb_op(mat, q):
  return mat @ q

@njit(parallel = True)
def DensityMatrix(newindex_, labels_, As_, Nsites):
    rho_ = np.zeros((len(newindex_), len(newindex_)))

    for i in prange(len(newindex_)):
        for j in range(len(newindex_)):
            auxi = nb_op(As_[0][labels_[i][0]], As_[1][labels_[i][1]])
            for k in range(1, Nsites-1):
                auxi = nb_op(auxi, As_[k+1][labels_[i][k+1]])

            auxj = nb_op(As_[0][labels_[j][0]], As_[1][labels_[j][1]])
            for k in range(1, Nsites-1):
                auxj = nb_op(auxj, As_[k+1][labels_[j][k+1]])
            rho_[i][j] = auxi[0][0]*np.conjugate(auxj[0][0])
    return rho_