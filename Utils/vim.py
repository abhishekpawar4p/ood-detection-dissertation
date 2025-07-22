import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA

class ViM:
    def __init__(self, model, layer_name="avgpool"):
        self.model = model.eval()
        self.layer_name = layer_name
        self.features = []
        self.logits = []
        self.pca = None
        self.U = None
        self.alpha = None
        self.hook = None

    def _hook_layer(self):
        def forward_hook(module, input, output):
            self.features.append(output.view(output.size(0), -1))

        layer = dict([*self.model.named_modules()])[self.layer_name]
        self.hook = layer.register_forward_hook(forward_hook)

    def extract_features(self, dataloader, device):
        self.features = []
        self.logits = []
        self._hook_layer()
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                logits = self.model(images)
                self.logits.append(logits.cpu())
        if self.hook:
            self.hook.remove()
        features = torch.cat(self.features, dim=0)
        logits = torch.cat(self.logits, dim=0)
        return features.numpy(), logits.numpy()

    def fit(self, features):
        pca = PCA()
        pca.fit(features)
        # Use all but last PC as null-space
        self.U = pca.components_[:-1].T
        self.pca = pca

    def compute_alpha(self, features, logits):
        V_logit = np.linalg.norm(features @ self.U, axis=1)
        max_logits = np.max(logits, axis=1)
        self.alpha = np.cov(V_logit, max_logits)[0, 1] / np.var(V_logit)

    def get_vim_score(self, features, logits):
        V_logit = np.linalg.norm(features @ self.U, axis=1)
        max_logits = np.max(logits, axis=1)
        score = -1 * (V_logit * self.alpha + max_logits)  # Negative because lower = OOD
        return score
