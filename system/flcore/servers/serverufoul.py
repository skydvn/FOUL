# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import time
from flcore.clients.clientufoul import clientUFOUL
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

class UFOUL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientUFOUL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        """ Learn-Unlearn Round Split """
        self.learn_round = args.learn_round
        self.unlearn_round = args.global_rounds - self.learn_round
        if self.unlearn_round < 0:
            raise ValueError("The global rounds must be higher than learn round")

        # self.load_model()
        self.Budget = []
        self.update_grads = None
        self.foul_c = args.c_parameter
        self.foul_rounds = args.cagrad_rounds
        self.foul_lr = args.cagrad_learning_rate
        self.momentum = args.momentum
        self.step_size = args.step_size
        self.meta_lr = args.meta_lr
        self.gamma = args.gamma
        self.beta = args.beta_foul
        self.device = args.device
        self.fgrad_balance = args.forget_balance
        self.rgrad_balance = args.retain_balance

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
            self.set_new_clients(clientUFOUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def unlearn(self):
        """ Need to put the fed unlearning part here which probably has a similar structure to the above learning part
        need to copy that code and configure it

        create a separate function for the unlearning part here

        for client in self.selected_clients:    ## (Question) in the unlearn function we need to set the clients to
                                                ##  unlearn ---> but when it comes to client unlearn what data we
                                                ##  need to specify right ??????
                client.unlearn()
        """
        ### Init Everything
        for client in self.clients:
            client.re_init(self.args)

        ### Fed Client wise forgetting stage
        print("\n======================================")
        print("\nFED Unlearning Stage")
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
            # self.receive_grads()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            old_model = copy.deepcopy(self.global_model)

            # self.aggregate_parameters(
            # self.aggregate_foul()
            meta_weights = self.aggregate_foul(
                meta_weights=self.global_model,
                selected_clients=self.uploaded_models,
                selected_id=self.uploaded_ids,
                lr_meta=self.meta_lr
            )
            self.global_model.load_state_dict(copy.deepcopy(meta_weights))
            # self.network.reset_weights(meta_weights)

            """
            Metrics of gradient angles here
            - Cosines between (gradients of retain set) vs. (aggregated gradient) 
            """
            r_angle_dict = {}
            f_angle_dict = {}
            # print(self.forget_list)
            for client in self.selected_clients:
                cos = self.cos_sim(old_model, self.global_model, client.model)
                norm = self.diff_weight(old_model, client.model)
                if client.id in self.forget_list:
                    f_angle_dict[f"{client.id}"] = cos
                else:
                    r_angle_dict[f"{client.id}"] = cos
                if self.args.log:
                    self.writer.add_scalar(f"client-charts/client{client.id}_angle", cos, self.current_round)
                    wandb.log({f"client-charts/client{client.id}_angle": cos}, step=self.current_round)
                    self.writer.add_scalar(f"client-charts/client{client.id}_norm", norm, self.current_round)
                    wandb.log({f"client-charts/client{client.id}_norm": norm}, step=self.current_round)

            print(f"======= Client Angle =======")
            print(r_angle_dict)
            print(f_angle_dict)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(self.foul_c = args.c_parameter
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientUFOUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.FUL_evaluate()

    def aggregate_foul(self, meta_weights, selected_clients, selected_id, lr_meta):
        # Lấy tất cả parameter names
        param_names = [name for name, _ in meta_weights.named_parameters()]

        """
        Hyper-gradients calculation & merge
        - In this section, the all_domains_grad_tensor is arranged according to the selected_id
        """
        all_client_grads = []
        retain_grads = []
        forget_grads = []

        for i_client, client in zip(selected_id, selected_clients):
            client_grad = [torch.flatten(inner_param - meta_param) for inner_param, meta_param in
                                 zip(client.parameters(), meta_weights.parameters())]
            client_grad_vector = torch.cat(client_grad).cpu()
            # all_client_grads.append(client_grad_vector)
            if i_client in self.forget_list: # This part should be verified in the future to make it flexible to the data.
                forget_grads.append(client_grad_vector)
            else:
                retain_grads.append(client_grad_vector)

        """
        - Retain Grads normalization.
        """
        if self.rgrad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            retain_grad_norms = [torch.norm(grad) for grad in retain_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(retain_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in retain_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_retain_grads = [grad * scale for grad, scale in zip(retain_grads, scaling_factors)]

            if self.args.log:
                for i, client_grad in enumerate(balanced_retain_grads):
                    diff_norms = torch.norm(client_grad)
                    wandb.log({f"client-charts/client{i}_norm_balance":diff_norms}, step=self.current_round)

            # Step 4: Stack the balanced gradients into a tensor
            retain_grad_tensor = torch.stack(balanced_retain_grads).cpu()
        else:
            retain_grad_tensor = torch.stack(forget_grads).cpu()



        """
        - Forget Grads normalization.
        """
        if self.fgrad_balance:
            # Apply balancing
            # Step 1: Compute norms for each gradient vector
            forget_grad_norms = [torch.norm(grad) for grad in forget_grads]

            # Step 2: Determine scaling factors to balance the norms
            # Example: Scale all norms to a target value (e.g., the average norm)
            target_norm = torch.mean(torch.tensor(forget_grad_norms))
            scaling_factors = [target_norm / norm if norm > 0 else 1.0 for norm in forget_grad_norms]

            # Step 3: Scale gradient vectors
            balanced_forget_grads = [grad * scale for grad, scale in zip(forget_grads, scaling_factors)]

            if self.args.log:
                for i, client_id in enumerate(self.forget_list):
                    diff_norms = torch.norm(balanced_forget_grads[i])
                    wandb.log({f"client-charts/client{client_id}_norm_balance": diff_norms}, step=self.current_round)

            # Step 4: Stack the balanced gradients into a tensor
            forget_grad_tensor = torch.stack(balanced_forget_grads).cpu()
        else:
            forget_grad_tensor = torch.stack(forget_grads).cpu()

        # all_domains_grad_tensor = torch.stack(all_client_grads)
        foul_grad = self.foul_update(retain_grad_tensor, forget_grad_tensor, len(selected_clients))
        # foul_grad = self.foul_update(all_domains_grad_tensor, len(selected_clients))

        # """ Update Grad to Model Parameters """
        # print(f"foul_grad: {foul_grad.size()}")

        meta_weights_vector = parameters_to_vector(meta_weights.parameters())
        vector_to_parameters(meta_weights_vector + foul_grad * lr_meta, meta_weights.parameters())

        # Tạo và in ra ParamDict mới từ trạng thái cập nhật của meta_weights
        updated_meta_weights = ParamDict(meta_weights.state_dict())

        return updated_meta_weights

    def foul_update(self, retain_grad_vec, forget_grad_vec, num_clients):
        """
        retain_grad_vec <torch tensor>: [num_retain_clients, dim] // CUDA:0
        forget_grad_vec <torch tensor>: [num_forget_clients, dim] // CUDA:0
        === === === === === === === ===
        The optimization is applied on CPU to save the GPU utilization.            print(f"current learn: {self.current_learn}")

        """

        r_grads = retain_grad_vec
        f_grads = forget_grad_vec

        """ Get dim for averaging """
        r_dim = r_grads.size()[0]
        f_dim = f_grads.size()[0]

        """ Retain mean grads """
        GGr = r_grads.mm(r_grads.t()).cpu()
        scale_r = (torch.diag(GGr)+1e-4).sqrt().mean()
        GGr = GGr / scale_r.pow(2)
        Ggr = GGr.mean(1, keepdims=True)
        ggr = Ggr.mean(0, keepdims=True)

        """ Forget mean grads """
        GGf = f_grads.mm(f_grads.t()).cpu()
        scale_f = (torch.diag(GGf)+1e-4).sqrt().mean()
        GGf = GGf / scale_f.pow(2)
        Ggf = GGf.mean(1, keepdims=True)
        ggf = Ggf.mean(0, keepdims=True)

        """ Get mean all """
        gg = (ggr * r_dim + ggf * f_dim)/(r_dim + f_dim)

        """ Define optimization variables w """
        w = torch.zeros(num_clients, 1, requires_grad=True)
        if num_clients == 50:
            w_opt = torch.optim.SGD([w], lr=self.foul_lr, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=self.foul_lr, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.foul_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            """ Minimization objective function """

            obj = (
                   (ww[0:r_dim].t().mm(Ggr)
                    - self.beta * ww[r_dim:r_dim+f_dim].t().mm(Ggf))
                   + c * (torch.abs(
                            ww[0:r_dim].t().mm(GGr).mm(ww[0:r_dim])
                          - self.beta * ww[r_dim:r_dim+f_dim].t().mm(GGf).mm(ww[r_dim:r_dim+f_dim])
                          )+ 1e-4).sqrt()
                  )

            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (torch.abs(ww[0:r_dim].t().mm(GGr).mm(ww[0:r_dim])
                          - self.beta * ww[r_dim:r_dim+f_dim].t().mm(GGf).mm(ww[r_dim:r_dim+f_dim])) + 1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)

        g = (torch.cat((r_grads.cpu(),f_grads.cpu()), dim=0).mean(0)).view(-1, 1)
        g += lmbda * ((r_grads.cpu() * ww[0:r_dim]).sum(0)
                      - self.beta * (f_grads.cpu() * ww[r_dim:r_dim+f_dim]).sum(0)
                      ).view(-1, 1)
        g /= (1 + self.foul_c ** 2)

        return (g.squeeze()).cuda()
