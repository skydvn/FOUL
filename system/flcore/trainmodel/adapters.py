import torch
import torch.nn as nn
import copy

class SparseAdapter(nn.Module):
    """
    Sparse adapter for FUSED unlearning algorithm.
    Creates a sparse mask and trainable parameters for a given layer.
    """
    def __init__(self, layer_shape, sparsity_ratio=0.9, device='cuda'):
        super(SparseAdapter, self).__init__()
        self.sparsity_ratio = sparsity_ratio
        self.device = device
        
        # Create sparse mask (1 = keep, 0 = drop)
        self.register_buffer('mask', self._create_sparse_mask(layer_shape))
        
        # Initialize adapter parameters (will be trained)
        self.adapter_params = nn.Parameter(
            torch.zeros(layer_shape, device=device)
        )
        
    def _create_sparse_mask(self, shape):
        """Random sparse mask: keep (1-sparsity_ratio) of parameters"""
        mask = torch.rand(shape) > self.sparsity_ratio
        return mask.float().to(self.device)
    
    def forward(self, original_weight):
        """Apply adapter: new_weight = original_weight + masked_adapter"""
        return original_weight + (self.adapter_params * self.mask)
    
    def get_num_trainable_params(self):
        """Return number of active (non-zero) parameters"""
        return int(self.mask.sum().item())


class AdapterManager:
    """
    Manages adapters for multiple layers in the model.
    Handles creation, training, and merging of adapters.
    """
    def __init__(self, model, critical_layers, sparsity_ratio=0.9, device='cuda'):
        self.model = model
        self.critical_layers = critical_layers
        self.sparsity_ratio = sparsity_ratio
        self.device = device
        self.adapters = {}
        
    def create_adapters(self):
        """Create sparse adapters for critical layers"""
        for name, param in self.model.named_parameters():
            if self._is_critical_layer(name):
                self.adapters[name] = SparseAdapter(
                    param.shape, 
                    self.sparsity_ratio, 
                    self.device
                ).to(self.device)
        
        print(f"Created {len(self.adapters)} adapters for critical layers")
        return self.adapters
    
    def _is_critical_layer(self, layer_name):
        """Check if layer is in critical layers list"""
        for critical in self.critical_layers:
            if critical in layer_name:
                return True
        return False
    
    def apply_adapters(self):
        """Apply adapters to model (for inference/evaluation)"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.adapters:
                    adapted_weight = self.adapters[name](param)
                    param.copy_(adapted_weight)
    
    def merge_adapters(self):
        """Permanently merge adapters into model weights"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.adapters:
                    merged_weight = self.adapters[name](param)
                    param.copy_(merged_weight)
        
        print("Adapters merged into model permanently")
    
    def get_adapter_state_dict(self):
        """Get state dict of all adapters (for communication)"""
        return {name: adapter.state_dict() 
                for name, adapter in self.adapters.items()}
    
    def load_adapter_state_dict(self, state_dict):
        """Load adapter state dict from server"""
        for name, adapter_state in state_dict.items():
            if name in self.adapters:
                self.adapters[name].load_state_dict(adapter_state)
