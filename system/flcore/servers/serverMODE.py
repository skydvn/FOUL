
import time
from flcore.clients.clientconda import clientConda
from flcore.servers.serverbase import Server
from utils.model_utils import ParamDict
from threading import Thread
import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import copy
from torch.optim.lr_scheduler import StepLR
import numpy as np
import statistics

from torch.utils.tensorboard import SummaryWriter
import wandb
## implmentation o


class ServerMoDe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # Initialize degradation model as a copy of the global model with random parameters
        self.degradation_model = copy.deepcopy(self.global_model)
        for param in self.degradation_model.parameters():
            param.data.uniform_(-1, 1)

    def unlearn(self, forget_clients, lambda_momentum=0.5, r_de=5, r_max=8):
        """
        Implements the MoDe unlearning process.
        Args:
            forget_clients (list): List of client IDs to forget.
            lambda_momentum (float): Momentum coefficient for degradation.
            r_de (int): Number of degradation rounds.
            r_max (int): Total unlearning rounds.
        """
        M_de = copy.deepcopy(self.degradation_model)  # Degradation model
        M = self.global_model  # Pre-trained global model

        for r in range(r_max):
            if r < r_de:
                # Knowledge erasure phase: Train M_de on remaining clients
                self.selected_clients = [c for c in self.clients if c.id not in forget_clients]
                self.send_models(M_de)
                self.receive_models(M_de)
                self.aggregate_parameters(M_de)

                # Momentum degradation: Update M towards M_de
                for param_m, param_de in zip(M.parameters(), M_de.parameters()):
                    param_m.data = (1 - lambda_momentum) * param_m.data + lambda_momentum * param_de.data

            # Memory guidance phase: Fine-tune M on all clients
            self.selected_clients = self.clients
            self.send_models(M)
            if r >= r_de:
                # Send M_de to target clients for pseudo-label guidance
                for client in self.selected_clients:
                    if client.id in forget_clients:
                        client.set_degradation_model(M_de)
            self.receive_models(M)
            self.aggregate_parameters(M)

        self.global_model = Ms