import numpy as np
from scipy.linalg import svd
from itertools import product
from numba import njit, prange

#Cria o estado inicial como um tensor da forma
#|psi> = sum_{j_1...j_Ns} C_{j_1 ... j_Ns} |j_1...j_Ns>
def InitialState(states_, coeffs_, dim_):
    C_ = np.zeros(dim_) #tensor C_{j_1 j_2 ... j_Ns}
    for i in range(len(states_)):
        C_[tuple(states_[i, :])] = coeffs_[i] #Coeficientes da base no tensor
    return C_/np.sqrt(np.vdot(coeffs_, coeffs_)) #Normalizacao
                 
#Agrupa indices. Por exemplo, considere os indices j_1, j_2, j_3 e j_4. suponha que queiramos juntar j_2, j_3 e j_4 num unico indice
#esta funcao entao cria o indice k = (j_2 j_3 j_4), que vai de 1 a max(j_2)*max(j_3)*max(j_4)
def GroupIndex(indexes_):

    labels_ = []
    newindex_ = []
    count = 0
    for label in product(range(0, 2), repeat=len(indexes_)): 
        labels_.append(list(label))
        newindex_.append(count)

        count += 1

    return np.array(labels_), np.array(newindex_)

#Esta funcao agrupa os indices à esquerda com as respectivas bond dimension
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

#Transforma o tensor C_{j_1, j_2 ... j_Ns} numa matriz inicial
def Tensor2Matrix(Nsites_, indexes_, states_, coeffs_, dim_):
    C_ = InitialState(states_, coeffs_, dim_)
    labels_, newindex_ = GroupIndex(indexes_[-(Nsites_-1):, :])

    M_ = np.zeros((len(indexes_[0, :]), len(newindex_)))

    for i in range(len(indexes_[0])):
        for ii in range(len(newindex_)):
            M_[i, ii] = C_[(i,) + tuple(labels_[ii])]
    return M_

#Aplica decomposicao em valor singular (SVD)
def ApplySVD(matrix_):
    return svd(matrix_, full_matrices=False)

#Faz um "reshape" da matriz correspondente ao sitio informado
def SweepSite(Nsites_, site_, bonddim_, indexes_, matrix_, maxbd_):

        Right_indexes_ = indexes_[-(Nsites_-site_-1):, :]

        Rlabels_, k1_ = GroupIndex(Right_indexes_)
        if bonddim_ < maxbd_:
            Llabels_, k2_ = GroupLeftIndex(indexes_[(Nsites_-site_-1), :], bonddim_)
        else:
            Llabels_, k2_ = GroupLeftIndex(indexes_[(Nsites_-site_-1), :], maxbd_)
        M_ = np.zeros((len(k2_), len(k1_)))

        for i in range(len(k2_)):
            for ii in range(len(k1_)):

                M_[i, ii] = matrix_[Llabels_[i, 1], len(k1_)*Llabels_[i, 0]+k1_[ii]]

        return M_

#Opera na matriz inicial, transformando-a, da esquerda para a direita, num produto de matrizes 
def CalculatePreMatrices(Nsites_, indexes_, states_, coeffs_, dim_, maxbd_):
    M_ = Tensor2Matrix(Nsites_, indexes_, states_, coeffs_, dim_)
    U_, Sigma_, Vdag_ = ApplySVD(M_)

    Us_ = []
    SingVals_ = []
    as_ = []
    for i in range(1, Nsites_-1):
        U_, Sigma_, Vdag_ = ApplySVD(M_)

        a_ = len(Sigma_)
        M_ = SweepSite(Nsites_, i, a_, indexes_, Vdag_, maxbd_)

        Us_.append(U_)
        SingVals_.append(Sigma_)
        as_.append(a_)

    U_, Sigma_, Vdag_ = ApplySVD(M_)

    Us_.append(U_)
    Us_.append(Vdag_)
    SingVals_.append(Sigma_)
    print('Max bond dim: ', max(as_))

    return Us_, SingVals_

#Multiplicacao de matriz
@njit(fastmath=True)
def nb_op(mat, q):
  return mat @ q

#Contrai as matrizes com seus respectivos valores singulares. Isso também eh feito da esquerda para a direita
def ContractSVals(Us_, SingVals_):
    As_ = []
    for i in range(len(Us_)-1):
        As_.append(np.array(np.split(nb_op(Us_[i], np.diag(SingVals_[i])), 2)))
    As_.append(np.array(np.split(Us_[-1], 2, axis = 1)))

    return As_


#Constroi a matriz de densidade
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

if __name__ == '__main__':
    Nsites = 6 #Numero de sistios
    Nbase = Nsites #Numero de estados "base" para a construcao do estado desejado

    maxbd = 2

    dim = []
    for i in range(Nsites):
        dim.append(2)#Dimensao do espaco de Hilbert local

    states = np.zeros((Nbase, Nsites), dtype = int) #Inicializacao dos estados
    coeffs = np.ones(Nbase) #Inicializacao dos coeficientes dos estados "base"
    for i in range(Nbase):
        states[i, i] = 1 #Aqui consideramos que o i-ésimo estado tem spin up no i-ésimo sitio, e todos os outros sao spin down
        coeffs[i] = 1 #Aqui consideramos que todos os coeficientes sao iguais

    dim = tuple(dim)

    #Valores para os indices
    js = np.zeros((Nsites, 2))
    js[:, 1] = 1

    Us, SingVals = CalculatePreMatrices(Nsites, js, states, coeffs, dim, maxbd) #Calculo dos Us e dos valores singulares

    As = ContractSVals(Us, SingVals) #Contracao dos Us com os valores singulares

    #Imprime as matrizes
    for i in range(len(As)):
        print(f'Sitio {i+1}:')
        print('Matriz indice 0: \n', np.round(As[i][0], 3), '\n')
        print('Matriz indice 1: \n', np.round(As[i][1], 3), '\n\n')
