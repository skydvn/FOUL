##implementation of the paper titled  - "CONDA: FAST FEDERATED UNLEARNING WITH CONTRIBUTION DAMPENING"
## https://arxiv.org/pdf/2410.04144

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
## implmentation of the server conda algo
class CONDA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientConda)

        self.dampening_constant = args.dampening_constant  # lambda 10 for MNIST, 1 for CIFAR-10 and CIFAR100
        self.cutoff_alpha = args.cutoff_alpha  # alpha para
        self.dampening_upper_bound = args.dampening_upper_bound  # U -- as per the paper for MNIST 10 and 1 for cifar10 and 100
        self.forget_list = [args.f_index * 5 + i for i in range(5)]  # need to get this from the args todo setup

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        """ Learn-Unlearn Round Split """
        self.learn_round = args.learn_round
        self.unlearn_round = args.global_rounds - self.learn_round
        if self.unlearn_round < 0:
            raise ValueError("The global rounds must be higher than learn round")

        self.device = args.device
        self.Budget = []

    def train(self):
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

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            old_model = copy.deepcopy(self.global_model)
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
            f_angle_dict = {k: f_angle_dict[k] for k in sorted(f_angle_dict, reverse=True)}
            r_angle_dict = {k: r_angle_dict[k] for k in sorted(r_angle_dict, reverse=True)}
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
            self.set_new_clients(clientConda)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def unlearn(self):
        for i in range(self.unlearn_round + 1):
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

            self.receive_models()

            ## parametters in CONDA paper
            all_gradients = []
            forget_gradients = []

            for client in self.selected_clients:
                gradient = []
                for server_param, client_param in zip(self.global_model.parameters(), client.model.parameters()):
                    gradient.append(client_param.data - server_param.data)  ## grad difference in equation

                all_gradients.append(gradient)  ## then looping through all clients
                ##forget clients update
                if client.id in self.forget_list:
                    forget_gradients.append(gradient)

            ## SSD stuff (selective synaptic damping ( i dont know why they use this name instead of parameter dampening))
            ratio = self.compute_conda_ratio(all_gradients, forget_gradients)
            zeta = self.dampening_constant * ratio
            beta = min(zeta, self.dampening_upper_bound)

            print(f"CONDA - Round {i}: ratio = {ratio:.4f}, dampening = {beta:.4f}")

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        #print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.FUL_evaluate()

    def compute_conda_ratio(self, all_gradients, forget_gradients):
        """Compute the ratio of gradient norms for CONDA dampening"""
        if not forget_gradients:
            return 1.0

        all_avg = sum(all_gradients) / len(all_gradients)

        forget_avg = sum(forget_gradients) / len(forget_gradients)

        ratio = torch.norm(all_avg) / (torch.norm(forget_avg) + 1e-8)
        return ratio