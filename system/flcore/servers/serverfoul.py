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
from flcore.clients.clientfoul import clientFOUL
from flcore.servers.serverbase import Server
from utils.model_utils import ParamDict
from threading import Thread
import torch
import copy
from torch.optim.lr_scheduler import StepLR
import numpy as np
import statistics


class FOUL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFOUL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.update_grads = None
        self.cagrad_c = args.c_parameter
        self.cagrad_rounds = args.cagrad_rounds
        self.cagrad_learning_rate = args.cagrad_learning_rate
        self.momentum = args.momentum
        self.step_size = args.step_size
        self.gamma = args.gamma
        self.device = args.device
        model_origin = copy.deepcopy(args.model)

    def train(self):
        """Fed learning stage"""
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
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
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFOUL)
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
        ### Fed Client wise forgetting stage
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients, self.current_learn = self.unlearn_select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.receive_grads()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # self.aggregate_parameters()
            # self.aggregate_foul()
            meta_weights = self.aggregate_foul(
                meta_weights=self.global_model,
                selected_clients=self.selected_clients, # This one should be list of models
                lr_meta=self.meta_lr
            )
            self.global_model.load_state_dict(copy.deepcopy(meta_weights))
            # self.network.reset_weights(meta_weights)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(self.cagrad_c = args.c_parameter
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFOUL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def aggregate_foul(self, meta_weights, selected_clients, lr_meta):
        # Lấy tất cả parameter names
        param_names = [name for name, _ in meta_weights.named_parameters()]

        # Tính toán gradient chênh lệch cho mỗi domain
        domain_grad_diffs = []
        for i_domain in range(self.client):
            domain_grads = []
            for (clone_param, meta_param, name) in zip(selected_clients[i_domain].model.parameters(),
                                                       meta_weights.parameters(), param_names):
                domain_grads.append(torch.zeros_like(torch.flatten(meta_param)))
            domain_grad_diffs.append(torch.cat(domain_grads))

        all_domains_grad_tensor = torch.stack(domain_grad_diffs)
        # print(all_domains_grad_tensor)
        foul_grad = self.foul_update(all_domains_grad_tensor, self.num_domains)

        # Cập nhật trọng số meta
        meta_weights_vector = parameters_to_vector(meta_weights.parameters())
        vector_to_parameters(meta_weights_vector + foul_grad * lr_meta, meta_weights.parameters())

        # Tạo và in ra ParamDict mới từ trạng thái cập nhật của meta_weights
        updated_meta_weights = ParamDict(meta_weights.state_dict())

        return updated_meta_weights

    def foul_update(self, grad_vec, num_tasks):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec

        GG = grads.mm(grads.t()).cpu()
        scale = (torch.diag(GG)+1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = GG.mean(1, keepdims=True)
        gg = Gg.mean(0, keepdims=True)

        w = torch.zeros(num_tasks, 1, requires_grad=True)
        if num_tasks == 50:
            w_opt = torch.optim.SGD([w], lr=50, momentum=0.5)
        else:
            w_opt = torch.optim.SGD([w], lr=25, momentum=0.5)

        c = (gg+1e-4).sqrt() * self.cagrad_c

        w_best = None
        obj_best = np.inf
        for i in range(21):
            w_opt.zero_grad()
            ww = torch.softmax(w, 0)
            obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
            if obj.item() < obj_best:
                obj_best = obj.item()
                w_best = w.clone()
            if i < 20:
                obj.backward(retain_graph=True)
                w_opt.step()

        ww = torch.softmax(w_best, 0)
        gw_norm = (ww.t().mm(GG).mm(ww)+1e-4).sqrt()

        lmbda = c.view(-1) / (gw_norm+1e-4)
        g = ((1/num_tasks + ww * lmbda).view(
            -1, 1).to(grads.device) * grads).sum(0) / (1 + self.cagrad_c**2)
        return g

#     def aggregate_foul(self):
#         grad_ez = sum(p.numel() for p in self.global_model.parameters())
#         grads = torch.Tensor(grad_ez, self.num_clients)
#
#         for index, model in enumerate(self.grads):
#             grad2vec2(model, grads, index)
#
#         # Split to grads retain & grads forget
#         g = self.cagrad(grads, self.num_clients)
#
#         model_origin = copy.deepcopy(self.global_model)
#         self.overwrite_grad2(self.global_model, g)
#         for param in self.global_model.parameters():
#             param.data += param.grad
#
#         angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.grads]
#         self.angle_value = statistics.mean(angle)
#
#         angle_value = []
#         for i in self.grads:
#             for j in self.grads:
#                 angle_value = [self.cosine_similarity(i, j)]
#
#         self.grads_angle_value = statistics.mean(angle_value)
#
#     def cagrad(self, grad_vec, num_tasks):
#
#         grads = grad_vec.to(self.device)
#
#         GG = grads.t().mm(grads)
#         # to(device)
#         scale = (torch.diag(GG) + 1e-4).sqrt().mean()
#         GG = GG / scale.pow(2)
#         Gg = GG.mean(1, keepdims=True)
#         gg = Gg.mean(0, keepdims=True)
#
#         w = torch.zeros(num_tasks, 1, requires_grad=True, device=self.device)
#         #         w = torch.zeros(num_tasks, 1, requires_grad=True).to(self.device)
#
#         if num_tasks == 50:
#             w_opt = torch.optim.SGD([w], lr=self.cagrad_learning_rate * 2, momentum=self.momentum)
#         else:
#             w_opt = torch.optim.SGD([w], lr=self.cagrad_learning_rate, momentum=self.momentum)
#
#         scheduler = StepLR(w_opt, step_size=self.step_size, gamma=self.gamma)
#
#         c = (gg + 1e-4).sqrt() * self.cagrad_c
#         w_best = None
#         obj_best = np.inf
#         for i in range(self.cagrad_rounds + 1):
#             w_opt.zero_grad()
#             ww = torch.softmax(w, dim=0)
#             obj = ww.t().mm(Gg) + c * (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
#             if obj.item() < obj_best:
#                 obj_best = obj.item()
#                 w_best = w.clone()
#             if i < self.cagrad_rounds:
#                 obj.backward()
#                 w_opt.step()
#                 scheduler.step()
#
#                 # Check this scheduler. step()
#
#         ww = torch.softmax(w_best, dim=0)
#         gw_norm = (ww.t().mm(GG).mm(ww) + 1e-4).sqrt()
#
#         lmbda = c.view(-1) / (gw_norm + 1e-4)
#         g = ((1 / num_tasks + ww * lmbda).view(
#             -1, 1).to(grads.device) * grads.t()).sum(0) / (1 + self.cagrad_c ** 2)
#         return g
#
#         # def overwrite_grad(self, m, newgrad, grad_dims):
#         #     newgrad = newgrad * self.num_clients  # to match the sum loss
#         #     cnt = 0
#         #     for mm in m.shared_modules():
#         #         for param in mm.parameters():
#         #             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#         #             en = sum(grad_dims[:cnt + 1])
#         #             this_grad = newgrad[beg: en].contiguous().view(param.data.size())
#         #             param.grad = this_grad.data.clone()
#         #             cnt += 1
#
#     def overwrite_grad2(self, m, newgrad):
#         newgrad = newgrad * self.num_clients
#         for param in m.parameters():
#             # Get the number of elements in the current parameter
#             num_elements = param.numel()
#
#             # Extract a slice of new_params with the same number of elements
#             param_slice = newgrad[:num_elements]
#
#             # Reshape the slice to match the shape of the current parameter
#             param.grad = param_slice.view(param.data.size())
#
#             # Move to the next slice in new_params
#             newgrad = newgrad[num_elements:]
#
#     # def grad2vec(m, grads, grad_dims, task):
#     #     grads[:, task].fill_(0.0)
#     #     cnt = 0
#     #     for mm in m.shared_modules():
#     #         for p in mm.parameters():
#     #             grad_cur = p.data.detach().clone()
#     #             beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
#     #             en = sum(grad_dims[:cnt + 1])
#     #             grads[beg:en, task].copy_(grad_cur.data.view(-1))
#     #             cnt += 1
#
# def grad2vec2(m, grads, task):
#     grads[:, task].fill_(0.0)
#     all_params = torch.cat([param.detach().view(-1) for param in m.parameters()])
#     # print(all_params.size())
#     grads[:, task].copy_(all_params)