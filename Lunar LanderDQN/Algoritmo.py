import torch
import math
from matplotlib import pyplot as plt
import gym
import os
import numpy as np
from Agente import DuploDQNAgente
from RedeNeural import RedeNeural



nomeAmbiente = "LunarLander-v2"
pastaLogs = "LunarLanderLogs/"
Salvarintervalo = 200
treinado = True
agenteTreinado = "LunarLanderLogs\LunarLander-v2-8000.pth" # Arquivo pht 
epsTreinado = 10 #Reproduzir quantos episódios do agente treinado 
# criar o ambiente
ambiente = gym.make(nomeAmbiente, render_mode= "human" if treinado else None)

epsTreino = 8000

###### Parâmetros #####
formaEntrada = 8
espacoAcao = ambiente.action_space.n
seed = 8
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dispositivo)
tamanhoBuffer = 100000   
tamanhoBatch = 64 
gamma = 0.99         
taxaAprendizado =  0.0005       
intervaloAtualizacao = 4
epsInicio = 1    
epsFim = 0.01         
epsDecaimento = 0.99    
tau = 1e-3             
######################


agente = DuploDQNAgente(formaEntrada,
                        espacoAcao,
                        seed,
                        dispositivo,
                        tamanhoBuffer,
                        tamanhoBatch,
                        gamma,
                        taxaAprendizado,
                        intervaloAtualizacao,
                        RedeNeural,
                        tau
                                         )


if treinado:
    controle = torch.load(agenteTreinado)
    agente.rede_online.load_state_dict(controle['state_dict'])
    agente.rede_alvo.load_state_dict(controle['state_dict'])
    agente.otimizador.load_state_dict(controle['otimizador'])
    epsInicio = controle['epsilon']

epocaInicio = 0
pontuacoes = []

# epsilonPorEp = lambda x: epsInicio + (epsInicio - epsFim) * math.exp(-1. * x /epsDecaimento)

def treinar(epsTreino):

    epsilon = epsInicio   
    for episodio in range(epocaInicio + 1, epsTreino+1):

        pontuacao = 0
        # eps = epsilonPorEp(episodio)

        estado = ambiente.reset()
        duracaoEp = 0
        concluido = False
        while not concluido:
            acao = agente.interagir(estado, epsilon)
            proxEstado, recompensa, concluido, info = ambiente.step(acao)
            pontuacao += recompensa

            duracaoEp += 1
            

            agente.passo(estado, acao, recompensa, proxEstado, concluido)
            estado = proxEstado
            if concluido:
                break
        print(duracaoEp)
        pontuacoes.append(pontuacao)
        if episodio % Salvarintervalo == 0:
            log = {'epoca': episodio,'state_dict': agente.rede_online.state_dict(),'otimizador': agente.otimizador.state_dict(),
                     'epsilon': epsilon }
            torch.save(log, f"{pastaLogs}{nomeAmbiente}-{episodio}.pth")

        epsilon = max(epsFim, epsDecaimento*epsilon) # diminui epsilon
    

        print(f"Episódio: {episodio}\nPontuação: {pontuacao}\n")
        
        
    return pontuacoes
if not treinado:
    pontuacoes = treinar(epsTreino)
    
else:
# Visualizar agente treinado
    for episódios in range(epsTreinado):
        trere = 0
        concluido = False
        estado = ambiente.reset()
        while not concluido:
            # ambiente.render()
            acao = agente.interagir(estado, epsilon=0.02)
            proxEstado, recompensa, concluido, _ = ambiente.step(acao)
            estado = proxEstado
            if concluido:
                ambiente.reset()
                break 
            trere += recompensa
        print(trere)
    ambiente.close()