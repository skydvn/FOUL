from flcore.clients.clientbase import Client
import torch

class ClientMoDe(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.degradation_model = None

    def set_degradation_model(self, degradation_model):
        """Set the degradation model for pseudo-label generation."""
        self.degradation_model = degradation_model

    def train(self):
        """Train the model, using pseudo-labels from degradation model if applicable."""
        self.model.train()
        if self.degradation_model and self.id in self.args.forget_clients:
            for batch in self.trainloader:
                inputs, _ = batch
                with torch.no_grad():
                    pseudo_labels = self.degradation_model(inputs).argmax(dim=1)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, pseudo_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        else:
            super().train()