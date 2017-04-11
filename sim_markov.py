#!/usr/bin/env python3
# coding: utf-8

import numpy as np

# Dados do problema:
MTTF = 5500  # horas
MTTRc = 0.75  # horas

C1 = 0.85
C2 = 0.95

dT = 0.1  # deltaT = 6 minutos
l = 1/MTTF  # lambda
mic = 1/MTTRc

# Matrizes de transicao para os dois modelos.
# Cada elemento A(i, j) é a probabilidade da cadeia ir do estado i para o j.
# Isso sai do seu desenho do modelo de Markov.
###### ATENCAO: ESTA MATRIZ É A TRANSPOSTA DA QUE O PROF FAZ NA AULA!!!! ######
transicao_confiabilidade = np.array([
        [1-2*l*dT, l*(C1+C2)*dT,      2*l*dT-(C1+C2)*l*dT],
        [mic*dT,   1-(2*l*dT+mic*dT), 2*l*dT],
        [0,        0,                 1]
    ])

transicao_disponibilidade = np.array([
        [1-2*l*dT, l*(C1+C2)*dT,      2*l*dT-(C1+C2)*l*dT],
        [mic*dT,   1-(2*l*dT+mic*dT), 2*l*dT],
        [mic*dT,   0,                 1-mic*dT]
    ])


def distribuicao_estacionaria_simulacao(matriz_transicao, epsilon = 1e-8, retorna_historico=False):
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
    pi = np.zeros(len(matriz_transicao))
    pi[0] = 1

    historico = [pi]
    A = np.transpose(matriz_transicao)
    
    historico.append(A.dot(pi))
    n=1
    print("Simulando....", end="")
    while np.linalg.norm(historico[-1] - historico[-2]) > epsilon:
        n += 1
        prox_pi = np.linalg.matrix_power(A, n).dot(historico[0])
        historico.append(prox_pi) 
    print("Pronto!")
    return historico[-1] if not retorna_historico else (historico[-1], np.array(historico))


print("Modelo de Disponibilidade:")
pi = distribuicao_estacionaria_simulacao(transicao_disponibilidade)
disp = (1-pi[-1])
print("Disponibilidade assintótica: {}%".format(disp*100))

print("\nModelo de Confiabilidade:")
# A próxima linha vai demorar horrores.
pi, hist = distribuicao_estacionaria_simulacao(transicao_confiabilidade, retorna_historico=True)
MTTF_sistema = sum([(1-x)*dT for x in hist[:,-1]])
print("MTTF do sistema:", MTTF_sistema)