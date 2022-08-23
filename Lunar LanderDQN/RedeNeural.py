import torch
from torch import nn
import torch.nn.functional as F
import random
class RedeNeural(nn.Module):
    """Modelo do Ator"""

    def __init__(self, estado, espacoAcao):
        """Inicializa os parâmetros e cria as camadas
         Parâmetros
        =================
         estado (int): Dimensão de cada estado
         espacoAcao (int): Número de ações
        """
        super(RedeNeural, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(estado, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, espacoAcao)
        
    def forward(self, estado):
        """Constrói a Rede que dado um estado retornará uma ação """
        saida = self.fc1(estado)
        saida = F.relu(saida)
        saida = self.fc2(saida)
        saida = F.relu(saida)
        return self.fc3(saida)