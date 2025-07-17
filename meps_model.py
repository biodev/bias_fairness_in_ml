import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

class DeepNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(DeepNN, self).__init__()

        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(32, 1)  # Output layer (1 neuron for binary classifications

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
		
        return self.output_layer(x)

class LitDNN(L.LightningModule):
    def __init__(self, use_weights=False):
        super().__init__()
        self.model = DeepNN(81)
        self.use_weights = use_weights

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y, weights = batch
        outputs = self.model(x)

        #pos_weight from here: https://discuss.pytorch.org/t/cross-entropy-loss-for-imbalanced-set-binary-classification/106554/2
        #Since there is around 5x more negative values, upweight the positive class by 5.0
        if self.use_weights == True:
            #adapted from https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/3
            loss = nn.BCEWithLogitsLoss(
			    pos_weight=torch.tensor(5.0, device=self.device),
			    reduction='none'
			)(outputs, y)
            
            loss = (loss * weights).mean()
        else:
            loss = nn.BCEWithLogitsLoss(
			    pos_weight=torch.tensor(5.0, device=self.device)
			)(outputs, y)
			
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, weights = batch
        outputs = self.model(x)
        val_loss = nn.BCEWithLogitsLoss(
			    pos_weight=torch.tensor(5.0, device=self.device)
			)(outputs, y)
        self.log("val_loss", val_loss)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, weights = batch
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
        return optimizer

class WeightedLitDNN(LitDNN):
    def __init__(self):
        super().__init__(use_weights=True)

class UnWeightedLitDNN(LitDNN):
    def __init__(self):
        super().__init__(use_weights=False)
