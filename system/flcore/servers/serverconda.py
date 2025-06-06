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

## implementation of the server conda algo
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

        print(f"round learn: {self.learn_round} | round unlearn: {self.unlearn_round}")

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
                print(f"\n------------- Round number: {i} -------------")
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

            if self.args.log:
                wandb.log({f"charts/zeta": 0}, step=self.current_round)
                wandb.log({f"charts/dampening": 0}, step=self.current_round)

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
                print(f"\n------------- Round number: {i} -------------")
                print("\nEvaluate global model")
                self.FUL_evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()

            ## parameters in CONDA paper
            all_gradients = []
            forget_gradients = []

            # TODO Create Undampening Blah Blah
            # get client-wise contribution for current iteration.
            client_contributions = get_client_contribution()

            # Add in the contributions for each client
            for client_id, contributions in client_contributions.items():

                # If the contributions doesn't exist for the client_id, make an empty dict.
                if avg_contributions.get(client_id) is None:
                    avg_contributions[client_id] = dict()

                # Iterate through each param and add in the contributions
                for param in start_model.keys():
                    # If the param id doesn't exists, add a zero vector.
                    if avg_contributions[client_id].get(param) is None:
                        avg_contributions[client_id][param] = torch.zeros_like(contributions[param])
                    avg_contributions[client_id][param] += contributions[param]

            ## SSD stuff (selective synaptic damping (i dont know why they use this name instead of parameter dampening))
            ratio = self.compute_conda_ratio(all_gradients, forget_gradients)
            zeta = self.dampening_constant * ratio
            beta = min(zeta, self.dampening_upper_bound)

            print(f"CONDA - Round {i}: ratio = {ratio:.4f}, dampening = {beta:.4f}")

            if self.args.log:
                wandb.log({f"charts/zeta": zeta}, step=self.current_round)
                wandb.log({f"charts/dampening": beta}, step=self.current_round)

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

    def get_client_contribution(self, start_model: OrderedDict, weights_path: str):
        checkpoint_list = os.listdir(weights_path)
        checkpoint_list = [checkpoint for checkpoint in checkpoint_list if
                           checkpoint.startswith("client_") and checkpoint.endswith(".pth")]
        checkpoint_list = [int(checkpoint[7:-4]) for checkpoint in checkpoint_list]
        checkpoint_list.sort()

        client_wise_differences = dict()
        for client_id in checkpoint_list:
            client_weights = torch.load(os.path.join(weights_path, f"client_{client_id}.pth"), map_location='cpu')
            difference = dict()
            for param in start_model.keys():
                difference[param] = torch.abs(start_model[param] - client_weights[param])
            client_wise_differences[client_id] = difference

        return client_wise_differences

    def get_group_contribution(self, contributions):
        """
        Get the average contribution of the group.
        """
        avg_contributions = dict()
        for key in contributions[0].keys():
            avg_contributions[key] = torch.zeros_like(contributions[0][key])
            for i in range(len(contributions)):
                avg_contributions[key] += contributions[i][key]
            avg_contributions[key] = torch.div(avg_contributions[key], len(contributions))
        return avg_contributions

    def apply_dampening(self, forget_client_contributions, retain_clients_contirbutions, dampening_constant,
                        dampening_upper_bound, ratio_cutoff):
        """
        Apply dampening to the global model based on the gradients of the local models.
        Args:
            global_model: The global model which will be dampened.
            forget_client_contributions: The gradient contributions of the forget clients/models.
            retain_clients_contributions: The gradient contributions of the retain clients/models.
            dampening_constant: The dampening constant.
            dampening_upper_bound: The upper bound for the final dampening factor. Used to cap the increasing of
            the parameters.
            ratio_cutoff: The cutoff/filter factor for ratios. Any parameter having the ratio greater than this value will not be updated.
              A high ratio means less contribution of the forget model, leading to less dampening.
        Returns:
            The updated global model.
        """

        with torch.no_grad():
            for (global_name, forget_grads), (index, retain_grads) in zip(
                    forget_client_contributions.items(),
                    retain_clients_contirbutions.items()
            ):

                if len(forget_grads.shape) > 0:
                    # Synapse Dampening with parameter dampening constant
                    weight = global_model[global_name]
                    # diff = torch.abs(g2_grads - g1_grads) # torch.abs(torch.abs(g2_grads) - torch.abs(g1_grads))
                    retain_contribution = torch.abs(retain_grads)  # epsilon
                    forget_contribution = torch.abs(forget_grads)
                    ratio = retain_contribution / forget_contribution
                    update_locations = (ratio < ratio_cutoff)
                    dampening_factor = torch.mul(ratio, dampening_constant)

                    update = dampening_factor[update_locations]
                    # Bound by 1 to prevent parameter values to increase.
                    min_locs = (update > dampening_upper_bound)
                    update[min_locs] = dampening_upper_bound
                    weight[update_locations] = weight[update_locations].mul(update)
        return global_model