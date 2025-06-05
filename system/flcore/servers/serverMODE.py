
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
## implementation

class ServerMoDe(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # Initialize degradation model as a copy of the global model with random parameters
        self.degradation_model = copy.deepcopy(self.global_model)
        for param in self.degradation_model.parameters():
            param.data.uniform_(-1, 1)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRetrain)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        """ Learn-Unlearn Round Split """
        self.learn_round = args.learn_round
        self.unlearn_round = args.global_rounds - self.learn_round
        # self.load_model()
        self.Budget = []

        self.lambda_momentum = 0.5
        self.r_de = 5
        self.r_max = 8

    def train(self):
        """Fed learning stage"""
        print("\n======================================")
        print("\nFED Learning Stage")
        for i in range(self.learn_round + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.FUL_evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            old_model = copy.deepcopy(self.global_model)

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            """
            Metrics of gradient angles here
            - Cosines between (gradients of retain set) vs. (aggregated gradient) 
            """
            r_angle_dict = {}
            f_angle_dict = {}
            # print(self.forget_list)
            for client in self.selected_clients:
                cos = self.cos_sim(old_model, self.global_model, client.model)
                if client.id in self.forget_list:
                    f_angle_dict[f"{client.id}"] = cos
                else:
                    r_angle_dict[f"{client.id}"] = cos
                if self.args.log:
                    self.writer.add_scalar(f"client-charts/client{client.id}_angle", cos, self.current_round)
                    wandb.log({f"client-charts/client{client.id}_angle": cos}, step=self.current_round)

            print(f"======= Client Angle =======")
            print(r_angle_dict)
            print(f_angle_dict)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRetrain)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def unlearn(self, forget_clients):
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