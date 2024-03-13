import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        in_features = 2
        hidden_dim = 100
        self.model = nn.Sequential(*[
            nn.Linear(in_features=in_features, out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_dim, out_features=3, bias=False),
            nn.Softmax(),
        ])

    def forward(self, x):
        return self.model(x).squeeze(-1)


class LitMLP(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(input=y_hat, target=y)
        return loss
