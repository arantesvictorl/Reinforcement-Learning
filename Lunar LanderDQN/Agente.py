import torch
import random
import torch.optim as optim
import numpy as np
from Buffer import BufferRepeticao
import torch.nn.functional as F
class DuploDQNAgente():
    def __init__(self, formaEntrada, espacoAcao, seed, dispositivo, tamanhoBuffer, tamanhoBatch, gamma, taxaAprendizado, intervaloAtualizacao, modelo, tau):
        """Inicializa o Agente do Double DQN
        
        Parâmetros
        =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            forma_entrada (tupla): dimensão de cada estado
            espacoAcao (int): quantas ações possíveis
            seed (int): seed aleatória
            dispositivo(string): se vai usar o CPU ou a GPU
            tamanhoBuffer (int): tamanho do buffer de repetição
            tamanhoBatch (int):  tamanho do batch
            gamma (float): taxa de desconto gamma
            taxaAprendizado (float): taxa de aprendizado 
            intervaloAtualizacao (int): qual a frequência para atualizar a rede
            inicioRepeticao (int): Depois de qual repetição deve ser iniciada a memória
            modelo(DQN): Modelo pytorch

        =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        """
        self.formaEntrada = formaEntrada
        self.espacoAcao = espacoAcao
        self.seed = seed
        self.dispositivo = dispositivo
        self.tamanhoBuffer = tamanhoBuffer
        self.tamanhoBatch = tamanhoBatch
        self.gamma = gamma
        self.taxaAprendizado = taxaAprendizado
        self.intervaloAtualizacao = intervaloAtualizacao
        self.DQN = modelo
        self.tau = tau

        # Rede-Q
        self.rede_online = self.DQN(formaEntrada, espacoAcao).to(self.dispositivo)
        self.rede_alvo = self.DQN(formaEntrada, espacoAcao).to(self.dispositivo)
        self.otimizador = optim.Adam(self.rede_online.parameters(), lr=self.taxaAprendizado)
        # Memória de Repetição
        self.memoria = BufferRepeticao(self.tamanhoBuffer, self.tamanhoBatch, self.seed, self.dispositivo)
        # Inicializa com passos de tempo como zero para contagem de atualização
        self.passos_tempo = 0

    def interagir(self, estado , epsilon=0):
        """Retorna ações para um determinado estado conforme a política atual"""

        estado = torch.from_numpy(estado).float().unsqueeze(0).to(self.dispositivo)
        self.rede_online.eval()
        with torch.no_grad():
            valores_acao = self.rede_online(estado)
        self.rede_online.train()

        # Seleção de ação por Epsilon-greedy
        if random.random() > epsilon:
            return np.argmax(valores_acao.cpu().data.numpy())
        else: 
            return random.choice(np.arange(self.espacoAcao))

    def passo(self, estado, acao, recompensa, proxEstado, concluido):

        # Salva a experiência na memória de repetição
        self.memoria.adiciona(estado, acao, recompensa, proxEstado, concluido)

        # Aprende a cada x passos de tempo de acordo com "intervaloAtualização"
        self.passos_tempo = (self.passos_tempo+1) % self.intervaloAtualizacao
        if self.passos_tempo == 0:
            if len(self.memoria) > self.tamanhoBatch:
                experiencias = self.memoria.amostra()
                self.aprenda(experiencias)


    def aprenda(self, experiencias):
        estados, acoes, recompensas, proxEstados, concluidos = experiencias


        # Pega o máximo valor Q previsto dos próximos estados (da nossa rede alvo)
        QproxEstado = self.rede_alvo(proxEstados).detach().max(1)[0].unsqueeze(1)

        # Calcula alvos para os estados atuais
        Qalvos = recompensas + self.gamma * QproxEstado * (1-concluidos)

        # Pega os valores Q previstos da nossa rede online
        Qprevisto = self.rede_online(estados).gather(1, acoes)

        # Calcula a perda
        self.perda = F.mse_loss(Qprevisto, Qalvos)

        # Minimizar a perda
        self.otimizador.zero_grad()
        self.perda.backward()
        self.otimizador.step()

        #Atualiza nossa rede online
        self.atualizacaoSoftware(self.rede_online, self.rede_alvo, self.tau)
            

    def atualizacaoSoftware(self, onlineModelo, alvoModelo, tau):
        """Atualiza os parâmetros de uma rede.

        θ_alvo = τ*θ_online + (1 - τ)*θ_alvo

        Parâmetros
        ======
            onlineModelo (PyTorch model): de que rede os pesos vão ser copiados
            alvoModelo  (PyTorch model): para que rede os pesos vão ser copiados
            tau (float): parâmetro de interpolação"""

        for parametroAlvo, parametroonline in zip(alvoModelo.parameters(), onlineModelo.parameters()):
            parametroAlvo.data.copy_(tau*parametroonline.data + (1.0-tau)*parametroAlvo.data)

        
        



