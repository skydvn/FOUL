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

import copy
import torch
import numpy as np
import time
from itertools import chain
from flcore.clients.clientbase import Client

class clientUFOUL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        # TODO L_inv + L_var
        # TODO Optimizers for Proto-Enc + NonProto-Enc
        self.proto_optimizer = torch.optim.SGD(
            chain(self.model.inv_encoder.parameters(),
                  self.model.var_encoder.parameters()
                  ),
            lr=self.learning_rate
        )
        self.proto_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.proto_optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        # TODO L_cls
        # TODO Optimizers for Shared-Enc + Proto-Enc + NonProto-Enc + Aux-Head
        self.cls_optimizer = torch.optim.SGD(
            chain(
                self.model.encoder.parameters(),
                self.model.inv_encoder.parameters(),
                self.model.var_encoder.parameters(), # FIXME Do we need to optimize this
                self.model.aux_head.parameters(),
                ),
            lr=self.learning_rate
        )
        self.cls_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.cls_optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        # TODO L_adv
        # TODO Optimizers for Shared-Enc + Proto-Enc + NonProto-Enc + Adv-Head
        self.adv_optimizer = torch.optim.SGD(
            chain(
                self.model.encoder.parameters(),     # FIXME Do we need to optimize this
                self.model.inv_encoder.parameters(),
                self.model.var_encoder.parameters(), # FIXME Do we need to optimize this
                self.model.adv_head.parameters(),
                ),
            lr=self.learning_rate
        )
        self.adv_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.adv_optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        # TODO L_rec
        # TODO Optimizers for Shared-Enc + Proto-Enc + NonProto-Enc + Aux-Dec
        self.meaning_optimizer = torch.optim.SGD(
            chain(
                self.model.encoder.parameters(),
                self.model.inv_encoder.parameters(),
                self.model.var_encoder.parameters(),
                self.model.aux_decoder.parameters(),
                ),
            lr=self.learning_rate
        )
        self.meaning_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.meaning_optimizer,
            gamma=args.learning_rate_decay_gamma
        )

        self.learning_rate_decay = args.learning_rate_decay

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # """
                # - measure each loss separately
                #
                # """
                # self.optimizer1.zero_grad()
                # loss1.backward()
                # self.optimizer1.step()
                # self.optimizer2.zero_grad()
                # loss2.backward()
                # self.optimizer2.step()
                # self.optimizer3.zero_grad()
                # loss3.backward()
                # self.optimizer3.step()
                # self.optimizer4.zero_grad()
                # loss4.backward()
                # self.optimizer4.step()
                # self.optimizer5.zero_grad()
                # loss5.backward()
                # self.optimizer5.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def unlearn(self):
        """ need to define a client unlearn method for the algorithm too
        which will be called in the serveravg train loop and initiate the unlearn method
        therefore we define the unlean method here"""
        pass
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
