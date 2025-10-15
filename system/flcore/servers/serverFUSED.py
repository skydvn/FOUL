import torch
import torch.nn as nn
import numpy as np
import copy
import wandb
from flcore.servers.serverbase import Server
from flcore.clients.clientFUSED import clientFUSED
from flcore.trainmodel.adapters import AdapterManager
from threading import Thread

class FUSED(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # FUSED-specific parameters
        self.num_critical_layers = args.num_critical_layers
        self.sparsity_ratio = args.sparsity_ratio
        self.unlearning_clients = set(args.unlearning_clients)
        self.unlearning_type = args.unlearning_type
        self.forget_class = args.forget_class if hasattr(args, 'forget_class') else None
        
        # Threading support
        self.num_threads = getattr(args, 'num_threads', 0)
        
        # Critical layers will be identified during CLI phase
        self.critical_layer_names = []
        self.adapter_manager = None
        
        # Result storage
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        
        # Set clients
        self.set_slow_clients()
        self.set_clients(clientFUSED)
        
        # Initialize Wandb
        self.use_wandb = getattr(args, 'use_wandb', True)
        if self.use_wandb:
            self.init_wandb(args)
        
        print(f"\n{'='*50}")
        print(f"FUSED Initialization")
        print(f"  Unlearning type: {self.unlearning_type}")
        print(f"  Unlearning clients: {self.unlearning_clients}")
        print(f"  Num critical layers: {self.num_critical_layers}")
        print(f"  Sparsity ratio: {self.sparsity_ratio}")
        print(f"  Wandb logging: {'Enabled' if self.use_wandb else 'Disabled'}")
        print(f"{'='*50}\n")
    
    def init_wandb(self, args):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project="FUSED-Federated-Unlearning",
            name=f"{self.dataset}_{self.algorithm}_run{self.times}",
            config={
                "algorithm": self.algorithm,
                "dataset": self.dataset,
                "num_clients": self.num_clients,
                "num_classes": self.num_classes,
                "global_rounds": self.global_rounds,
                "local_epochs": self.local_epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "num_critical_layers": self.num_critical_layers,
                "sparsity_ratio": self.sparsity_ratio,
                "unlearning_type": self.unlearning_type,
                "unlearning_clients": list(self.unlearning_clients),
                "model": args.model.__class__.__name__,
                "join_ratio": self.join_ratio,
            },
            tags=["FUSED", "unlearning", self.dataset],
            notes=f"Domain unlearning on {self.dataset} dataset"
        )
        
        # Log model architecture
        if self.use_wandb:
            wandb.watch(self.global_model, log="all", log_freq=10)
    
    def train(self):
        """Main training pipeline for FUSED with wandb logging"""
        # Phase 1: Normal Federated Learning (pre-unlearning)
        print("\n" + "="*60)
        print("PHASE 1: Normal Federated Learning (Pre-Unlearning)")
        print("="*60)
        
        if self.use_wandb:
            wandb.log({"phase": 1, "phase_name": "Pre-Unlearning Training"})
        
        for round_idx in range(self.global_rounds // 2):
            self.selected_clients = self.select_clients()
            self.send_models()
            
            # Train clients
            for client in self.selected_clients:
                client.train()
            
            self.receive_models()
            self.aggregate_parameters()
            
            # Evaluate and log
            if round_idx % self.eval_gap == 0:
                print(f"\n[Round {round_idx}] Pre-unlearning Evaluation:")
                self.evaluate(round_idx, phase="pre-unlearning")
        
        # Save pre-unlearning model
        self.original_model = copy.deepcopy(self.global_model)
        
        # Phase 2: Critical Layer Identification
        print("\n" + "="*60)
        print("PHASE 2: Critical Layer Identification (CLI)")
        print("="*60)
        
        if self.use_wandb:
            wandb.log({"phase": 2, "phase_name": "Critical Layer Identification"})
        
        self.critical_layer_identification()
        
        # Phase 3: Construct Sparse Adapters
        print("\n" + "="*60)
        print("PHASE 3: Sparse Adapter Construction")
        print("="*60)
        
        if self.use_wandb:
            wandb.log({"phase": 3, "phase_name": "Sparse Adapter Construction"})
        
        self.construct_sparse_adapters()
        
        # Phase 4: Unlearning via Adapter Training
        print("\n" + "="*60)
        print("PHASE 4: Unlearning via Sparse Adapters")
        print("="*60)
        
        if self.use_wandb:
            wandb.log({"phase": 4, "phase_name": "Unlearning via Adapters"})
        
        for round_idx in range(self.global_rounds // 2, self.global_rounds):
            self.selected_clients = self.select_remaining_clients()
            self.send_adapters()
            
            # Train only adapters
            for client in self.selected_clients:
                client.train_adapters()
            
            self.receive_adapters()
            self.aggregate_adapters()
            
            if round_idx % self.eval_gap == 0:
                print(f"\n[Round {round_idx}] Unlearning Evaluation:")
                self.evaluate_unlearning(round_idx)
        
        # Phase 5: Merge adapters
        print("\n" + "="*60)
        print("PHASE 5: Merging Adapters")
        print("="*60)
        
        if self.use_wandb:
            wandb.log({"phase": 5, "phase_name": "Adapter Merging"})
        
        self.merge_adapters()
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        ra, fa = self.evaluate_unlearning(self.global_rounds, final=True)
        
        # Log final summary
        if self.use_wandb:
            wandb.log({
                "final/remaining_accuracy": ra,
                "final/forgetting_accuracy": fa,
                "final/unlearning_effectiveness": ra - fa,
            })
            
            # Create summary table
            summary_table = wandb.Table(
                columns=["Metric", "Value"],
                data=[
                    ["Remaining Accuracy (RA)", f"{ra:.4f}"],
                    ["Forgetting Accuracy (FA)", f"{fa:.4f}"],
                    ["Unlearning Gap (RA-FA)", f"{ra-fa:.4f}"],
                    ["Critical Layers", len(self.critical_layer_names)],
                    ["Sparsity Ratio", self.sparsity_ratio],
                ]
            )
            wandb.log({"final/summary": summary_table})
        
        # Save results
        self.save_results()
        self.save_global_model()
        
        # Finish wandb run
        if self.use_wandb:
            wandb.finish()
    
    def evaluate(self, round_idx, phase="training"):
        """Evaluate and log results with wandb"""
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        
        # Calculate metrics
        test_acc = sum(stats[2]) / len(stats[2])
        test_auc = sum(stats[3]) / len(stats[3])
        train_loss = sum(stats_train[2]) / len(stats_train[2])
        std_acc = np.std(stats[2])
        std_auc = np.std(stats[3])
        
        # Save results
        self.rs_test_acc.append(test_acc)
        self.rs_test_auc.append(test_auc)
        self.rs_train_loss.append(train_loss)
        
        # Print
        print(f"Averaged Train Loss: {train_loss:.4f}")
        print(f"Averaged Test Accuracy: {test_acc:.4f}")
        print(f"Averaged Test AUC: {test_auc:.4f}")
        print(f"Std Test Accuracy: {std_acc:.4f}")
        print(f"Std Test AUC: {std_auc:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                f"{phase}/round": round_idx,
                f"{phase}/train_loss": train_loss,
                f"{phase}/test_accuracy": test_acc,
                f"{phase}/test_auc": test_auc,
                f"{phase}/test_accuracy_std": std_acc,
                f"{phase}/test_auc_std": std_auc,
            })
    
    def critical_layer_identification(self):
        """Identify critical layers with wandb visualization"""
        print("Running Critical Layer Identification...")
        
        self.selected_clients = self.clients
        self.send_models()
        
        for client in self.selected_clients:
            client.train()
        
        layer_diffs = self._calculate_layer_differences()
        
        sorted_layers = sorted(layer_diffs.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        self.critical_layer_names = [name for name, _ in 
                                     sorted_layers[:self.num_critical_layers]]
        
        print(f"\nCritical Layers Identified:")
        for i, (name, diff) in enumerate(sorted_layers[:self.num_critical_layers]):
            print(f"  {i+1}. {name}: {diff:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            # Create bar chart of layer sensitivities
            layer_names = [name for name, _ in sorted_layers[:20]]  # Top 20
            layer_values = [diff for _, diff in sorted_layers[:20]]
            
            wandb.log({
                "cli/layer_sensitivity": wandb.plot.bar(
                    wandb.Table(
                        data=[[name, val] for name, val in zip(layer_names, layer_values)],
                        columns=["Layer", "Sensitivity"]
                    ),
                    "Layer", "Sensitivity",
                    title="Top 20 Critical Layers by Sensitivity"
                ),
                "cli/num_critical_layers": len(self.critical_layer_names),
                "cli/critical_layers": self.critical_layer_names,
            })
        
        return self.critical_layer_names
    
    def _calculate_layer_differences(self):
        """Calculate Manhattan distance for each layer"""
        layer_diffs = {}
        original_params = {name: param.data.clone() 
                          for name, param in self.original_model.named_parameters()}
        
        total_samples = sum([c.train_samples for c in self.selected_clients])
        
        for client in self.selected_clients:
            weight = client.train_samples / total_samples
            
            for name, param in client.model.named_parameters():
                orig_param = original_params[name]
                diff = torch.sum(torch.abs(param.data - orig_param)).item()
                
                if name not in layer_diffs:
                    layer_diffs[name] = 0.0
                
                layer_diffs[name] += diff * weight
        
        return layer_diffs
    
    def construct_sparse_adapters(self):
        """Create sparse adapters with wandb logging"""
        self.adapter_manager = AdapterManager(
            self.global_model,
            self.critical_layer_names,
            self.sparsity_ratio,
            self.device
        )
        
        self.adapters = self.adapter_manager.create_adapters()
        
        total_adapter_params = sum([adapter.get_num_trainable_params() 
                                   for adapter in self.adapters.values()])
        
        total_model_params = sum([p.numel() for p in self.global_model.parameters()])
        
        print(f"\nAdapter Statistics:")
        print(f"  Total model parameters: {total_model_params:,}")
        print(f"  Total adapter parameters: {total_adapter_params:,}")
        print(f"  Ratio: {100*total_adapter_params/total_model_params:.2f}%")
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "adapters/total_model_params": total_model_params,
                "adapters/total_adapter_params": total_adapter_params,
                "adapters/adapter_ratio": total_adapter_params / total_model_params,
                "adapters/num_adapters": len(self.adapters),
                "adapters/sparsity_ratio": self.sparsity_ratio,
            })
    
    def select_remaining_clients(self):
        """Select only clients that should retain their data"""
        remaining_clients = [c for c in self.clients 
                           if c.id not in self.unlearning_clients]
        
        num_join = min(self.num_join_clients, len(remaining_clients))
        selected = np.random.choice(remaining_clients, num_join, replace=False)
        
        return selected
    
    def send_adapters(self):
        """Send adapters to selected clients"""
        for client in self.selected_clients:
            adapter_state = self.adapter_manager.get_adapter_state_dict()
            client.set_adapters(adapter_state, self.critical_layer_names)
    
    def receive_adapters(self):
        """Collect trained adapters from clients"""
        assert len(self.selected_clients) > 0
        
        self.uploaded_ids = []
        self.uploaded_adapters = []
        self.uploaded_weights = []
        
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_adapters.append(client.get_adapters())
            self.uploaded_weights.append(client.train_samples)
    
    def aggregate_adapters(self):
        """FedAvg aggregation for adapters"""
        total_samples = sum(self.uploaded_weights)
        
        for name in self.adapters.keys():
            aggregated_param = torch.zeros_like(
                self.adapters[name].adapter_params
            )
            
            for i, adapter_dict in enumerate(self.uploaded_adapters):
                weight = self.uploaded_weights[i] / total_samples
                aggregated_param += adapter_dict[name]['adapter_params'] * weight
            
            self.adapters[name].adapter_params.data = aggregated_param
    
    def merge_adapters(self):
        """Merge adapters with original model permanently"""
        self.adapter_manager.merge_adapters()
        print("Adapters successfully merged with global model")
    
    def evaluate_unlearning(self, round_idx, final=False):
        """Evaluate unlearning performance with wandb logging"""
        stats = self.test_metrics()
        
        remaining_acc = []
        forgetting_acc = []
        
        for i, client in enumerate(self.clients):
            if client.id in self.unlearning_clients:
                forgetting_acc.append(stats[2][i])
            else:
                remaining_acc.append(stats[2][i])
        
        ra = np.mean(remaining_acc) if remaining_acc else 0.0
        fa = np.mean(forgetting_acc) if forgetting_acc else 0.0
        
        print(f"\nUnlearning Metrics:")
        print(f"  Remaining Accuracy (RA): {ra:.4f}")
        print(f"  Forgetting Accuracy (FA): {fa:.4f}")
        print(f"  Unlearning Gap: {ra - fa:.4f}")
        
        # Log to wandb
        if self.use_wandb:
            prefix = "final" if final else "unlearning"
            wandb.log({
                f"{prefix}/round": round_idx,
                f"{prefix}/remaining_accuracy": ra,
                f"{prefix}/forgetting_accuracy": fa,
                f"{prefix}/unlearning_gap": ra - fa,
                f"{prefix}/remaining_accuracy_std": np.std(remaining_acc) if remaining_acc else 0,
                f"{prefix}/forgetting_accuracy_std": np.std(forgetting_acc) if forgetting_acc else 0,
            })
        
        return ra, fa
    
    def save_results(self):
        """Save training results to HDF5 file"""
        import h5py
        import os
        
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        
        algo = self.dataset + "_" + self.algorithm
        file_path = os.path.join(self.save_folder_name, "{}_{}_{}.h5".format(algo, self.goal, self.times))
        
        print(f"\nSaving results to {file_path}")
        
        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
            hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
            hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        
        print("Results saved successfully!")
    
    def save_global_model(self):
        """Save the final global model"""
        import os
        
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model_file = os.path.join(model_path, f"{self.algorithm}_{self.times}.pt")
        torch.save(self.global_model.state_dict(), model_file)
        print(f"Global model saved to {model_file}")
        
        # Log model as wandb artifact
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"model-{self.dataset}-{self.algorithm}",
                type="model",
                description=f"FUSED unlearned model for {self.dataset}"
            )
            artifact.add_file(model_file)
            wandb.log_artifact(artifact)
