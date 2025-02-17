import numpy as np

#importa as funções do arquivo MPS_Func
from MPS_Func import *

Nsites = 12 #Numero de sitios
Nbase = Nsites #Numero de estados "base" para a construcao do estado desejado
max_bond_dimension = 4
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

Us, SingVals = CalculatePreMatrices(Nsites, js, states, coeffs, dim, max_bond_dimension) #Calculo dos Us e dos valores singulares

As = ContractSVals(Us, SingVals) #Contracao dos Us com os valores singulares

#Imprime as matrizes
for i in range(len(As)):
    print(f'Sitio {i+1}:')
    print('Matriz indice 0: \n', np.round(As[i][0], 3), '\n')
    print('Matriz indice 1: \n', np.round(As[i][1], 3), '\n\n')

# labels_ = []
# newindex_ = []
# count = 0
# for label in product(range(0, 2), repeat=Nsites): 
#     labels_.append(list(label))
#     newindex_.append(count)

#     count += 1

# rho = DensityMatrix(np.array(newindex_), np.array(labels_), As, Nsites)

# print(np.max(np.linalg.eigvalsh(rho)))

# import matplotlib.pyplot as plt
# plt.imshow(rho[:100, :100])
# plt.colorbar()
# plt.show()