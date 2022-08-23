import torch
import numpy as np
import random
from collections import namedtuple, deque

class BufferRepeticao():
    def __init__(self, tamanhoBuffer, tamanhoBatch, seed, dispostivo) -> None:
        self.memoria = deque(maxlen=tamanhoBuffer)
        self.tamanhoBatch = tamanhoBatch
        self.experiencia = namedtuple("experiencia", field_names=["estado", "acao", "recompensa", "proxEstado", "concluido"])
        # self.seed = random.seed(seed)
        self.dispositivo = dispostivo


    def adiciona(self, estado, acao, recompensa, proxEstado, concluido):
        """Adiciona uma experiência ao Buffer"""
        xp = self.experiencia(estado, acao, recompensa, proxEstado, concluido)
        self.memoria.append(xp)

    def amostra(self):
        """Pega aleatóriamente algumas experiencias da memória"""
        experiencias = random.sample(self.memoria, k=self.tamanhoBatch)

        estados = torch.from_numpy(np.vstack([xp.estado for xp in experiencias if xp is not None])).float().to(self.dispositivo)
        acoes = torch.from_numpy(np.vstack([xp.acao for xp in experiencias if xp is not None])).long().to(self.dispositivo)
        recompensas = torch.from_numpy(np.vstack([xp.recompensa for xp in experiencias if xp is not None])).float().to(self.dispositivo)
        proxEstados = torch.from_numpy(np.vstack([xp.proxEstado for xp in experiencias if xp is not None])).float().to(self.dispositivo)
        concluidos = torch.from_numpy(np.vstack([xp.concluido for xp in experiencias if xp is not None])).float().to(self.dispositivo )

        return (estados, acoes, recompensas, proxEstados, concluidos)

    def __len__(self):
        """Retorna o número de elementos na memoria"""
        return len(self.memoria)