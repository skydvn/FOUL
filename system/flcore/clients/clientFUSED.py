import torch
import torch.nn as nn
import numpy as np
import copy
from flcore.clients.clientbase import Client
from flcore.trainmodel.adapters import AdapterManager

class clientFUSED(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # FUSED-specific attributes
        self.adapter_manager = None
        self.adapters = {}
        self.critical_layer_names = []
        self.sparsity_ratio = args.sparsity_ratio
        
        # Store original model weights for restoration
        self.original_weights = {}
        
        # Domain information (for PACS)
        self.domain_id = kwargs.get('domain_id', None)
        self.domain_name = kwargs.get('domain_name', None)
    
    def set_adapters(self, adapter_state_dict, critical_layer_names):
        """Receive adapters from server and set up local adapter manager"""
        self.critical_layer_names = critical_layer_names
        
        # Create adapter manager
        self.adapter_manager = AdapterManager(
            self.model,
            critical_layer_names,
            self.sparsity_ratio,
            self.device
        )
        
        # Create adapters
        self.adapters = self.adapter_manager.create_adapters()
        
        # Load state from server
        self.adapter_manager.load_adapter_state_dict(adapter_state_dict)
        
        # Store original weights for layers with adapters
        self._store_original_weights()
    
    def _store_original_weights(self):
        """Store original model weights before applying adapters"""
        self.original_weights = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.adapters:
                    self.original_weights[name] = param.data.clone().detach()
    
    def train(self):
        """Standard training for pre-unlearning phase (Phase 1)"""
        trainloader = self.load_train_data()
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
    
    def train_adapters(self):
        """
        Train only the sparse adapters while keeping original model frozen.
        This is the core unlearning step (Phase 4).
        """
        trainloader = self.load_train_data()
        
        # Store original model state
        original_model_state = {}
        for name, param in self.model.named_parameters():
            if name in self.adapters:
                original_model_state[name] = param.data.clone()
        
        # Create optimizer for adapters only
        adapter_params = []
        for adapter in self.adapters.values():
            adapter_params.append(adapter.adapter_params)
        
        optimizer = torch.optim.SGD(adapter_params, lr=self.learning_rate)
        
        self.model.train()
        
        for epoch in range(self.local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                optimizer.zero_grad()
                
                # Apply adapters during forward pass
                # This modifies model parameters temporarily
                self._apply_adapters_for_forward(original_model_state)
                
                # Forward pass
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Backward pass - gradients flow to adapter_params
                loss.backward()
                optimizer.step()
                
                # Restore original weights
                self._restore_original_weights(original_model_state)
    
    def _apply_adapters_for_forward(self, original_state):
        """
        Apply adapters during forward pass.
        Uses in-place operations to maintain gradient flow.
        """
        for name, param in self.model.named_parameters():
            if name in self.adapters:
                # Apply adapter: param = original + (adapter_params * mask)
                adapter = self.adapters[name]
                adapted_weight = original_state[name] + (adapter.adapter_params * adapter.mask)
                param.data.copy_(adapted_weight)
    
    def _restore_original_weights(self, original_state):
        """Restore original weights after forward pass"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.adapters:
                    param.data.copy_(original_state[name])
    
    def get_adapters(self):
        """Send adapter state dict to server"""
        return self.adapter_manager.get_adapter_state_dict()
