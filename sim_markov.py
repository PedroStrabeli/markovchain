#!/usr/bin/env python3
# coding: utf-8

import numpy as np

# Dados do problema:
MTTF = 6500  # horas (entre 4000 e 7500)
MTTRc = 1.75  # horas
MTTRp = 4.00  # horas

C = 0.85
C2 = 0.95

dT = 0.1  # deltaT = 6 minutos
l = 1/MTTF  # lambda
mic = 1/MTTRc

# Matrizes de transicao para os dois modelos.
# Cada elemento A(i, j) é a probabilidade da cadeia ir do estado i para o j.
# Isso sai do seu desenho do modelo de Markov.
###### ATENCAO: ESTA MATRIZ É A TRANSPOSTA DA QUE O PROF FAZ NA AULA!!!! ######
confiabilidade = np.array([
        [1-2*l*dT,     l*C*dT,         l*dT,      (1-C)*l*dT    ],
        [ mic*dT,   1-((l+mic)*dT),      0,         2*l*dT      ],
        [ mic*dT,        0            ,1-mic*dT,     l*dT       ],
        [    0,          0,              0,             1       ]
    ])

disponibilidade = np.array([
        [1-2*l*dT, l*C*dT,         l*dT,   (1-C)*l*dT],
        [mic*dT,   1-((l+mic)*dT),      0,      2*l*dT],
        [mic*dT,        0            ,1-mic*dT,     l*dT],
        [mic*dT, 0, 0, 1-mic*dT]
    ])


def simulacao(matriz, epsilon = 1e-8, historico=False):
     """ Encontra a distribuicao estacionaria para uma cadeia de markov.
    Utiliza o metodo de elevar a matriz de transicao a altas potencias ate convergir.
    
    Args:
        matriz_transicao (np.array): Matriz de transicao da cadeia de markov.
        epsilon (float): Quando o modulo da variacao do vetor distribuicao estacionaria entre duas
            iteracoes for menor que epsilon, o algoritmo convergiu.
        retorna_historico (bool): Indica se a funcao deve retornar tambem os vetores intermediarios.
    
    Returns:
        np.array: O vetor distribuicao estacionaria.
        np.array: Se 'retorna_historico' for True, todos os vetores intermediarios encontrados na simulacao.
    """
    pi = np.zeros(len(matriz))
    pi[0] = 1

    historico = [pi]
    A = np.transpose(matriz)
    
    historico.append(A.dot(pi))
    n=1
    print "Iniciando a simulação....",
    while np.linalg.norm(historico[-1] - historico[-2]) > epsilon:
        n += 1
        prox_pi = np.linalg.matrix_power(A, n).dot(historico[0])
        historico.append(prox_pi) 
    print("Simulação finalizada!")
    return historico[-1] if not historico else (historico[-1], np.array(historico))



print("\nModelo de Confiabilidade:")
# A próxima linha vai demorar horrores.
pi, hist = simulacao(confiabilidade, historico=True)
MTTF_sistema = sum([(1-x)*dT for x in hist[:,-1]])
print("MTTF do sistema:", MTTF_sistema)
print("Modelo de Disponibilidade:")
pi = simulacao(disponibilidade)
disp = (1-pi[-1])
print("Disponibilidade assintótica: {}%".format(disp*100))