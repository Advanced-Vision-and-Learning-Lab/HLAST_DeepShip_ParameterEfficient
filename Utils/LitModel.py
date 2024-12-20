import torch
from Utils.Network_functions import initialize_model
import torch.nn.functional as F
import lightning as L
import torchmetrics
    
class LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, numBins, RR):
        super().__init__()

        self.learning_rate = Params['lr']

        self.model_ft, self.feature_extraction_layer = initialize_model(model_name, num_classes,
                                                                        numBins,RR,Params['sample_rate'],
                                                                        segment_length=Params['segment_length'],
                                                                        window_length=Params['window_length'],
                                                                        hop_length=Params['hop_length'],
                                                                        number_mels=Params['number_mels'],
                                                                        t_mode=Params['train_mode'],
                                                                        h_shared=Params['histograms_shared'],
                                                                        a_shared=Params['adapters_shared'],
                                                                        parallel=Params['parallel'],
                                                                        input_feature=Params['feature'],
                                                                        adapter_location=Params['adapter_location'],
                                                                        adapter_mode=Params['adapter_mode'],
                                                                        histogram_location=Params['histogram_location'],
                                                                        histogram_mode=Params['histogram_mode'])

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()

    def forward(self, x):
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        return y_pred

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        loss = F.cross_entropy(y_pred, y)

        self.train_acc(y_pred, y)

        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        val_loss = F.cross_entropy(y_pred, y)

        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        return val_loss
 
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        
        test_loss = F.cross_entropy(y_pred, y)
        
        self.test_acc(y_pred, y)
    
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return test_loss

    def configure_optimizers(self):

        base_lr = self.learning_rate  
        optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr)

        return optimizer


