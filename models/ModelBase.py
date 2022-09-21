import torch
import torchmetrics
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import Dict, List

class ModelBase(pl.LightningModule):
    def __init__(self, metrics: Dict, classes: List[str]):
        super().__init__()
        self.save_hyperparameters(ignore='metrics')
        self.cm_metric = metrics.pop('CM', None)
        metrics = torchmetrics.MetricCollection(metrics)
        self.val_metrics = metrics.clone(f"val/")
        self.test_metrics = metrics.clone(f"test/")
        self.classes = classes
        self.example_input_array = torch.zeros(1,2,1024)

    def training_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        loss = self.loss(output, target)
        if self.global_step!= 0: self.logger.log_metrics({'train/loss': loss, 'epoch': self.current_epoch}, self.global_step)
        return loss

    def validation_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        self.val_metrics.update(output, target)
        if self.cm_metric: self.cm_metric.update(output, target)

    def validation_epoch_end(self, outputs):
        metrics_dict = self.val_metrics.compute()
        self.val_metrics.reset()
        if self.global_step!=0: self.logger.log_metrics(metrics_dict, self.global_step)
        
        if self.cm_metric:
            mpl.use("Agg")
            fig = plt.figure(figsize=(13, 13))
            cm = self.cm_metric.compute().cpu().detach().numpy()
            self.cm_metric.reset()
            ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self.classes, rotation=90)
            ax.yaxis.set_ticklabels(self.classes, rotation=0)
            plt.tight_layout()
            self.logger.experiment.add_figure("val/cm", fig, global_step=self.global_step)
        
    def test_step(self, batch, batch_nb):
        data, target = batch
        output = self.forward(data)
        self.test_metrics.update(output, target)
        if self.cm_metric: self.cm_metric.update(output, target)
        return {"test_out": output, "test_true": target}

    def test_epoch_end(self, outputs):
        metrics_dict = self.test_metrics.compute()
        self.test_metrics.reset()
        if self.global_step!= 0: self.logger.log_metrics(metrics_dict, self.global_step)
        
        if self.cm_metric:
            mpl.use("Agg")
            fig = plt.figure(figsize=(13, 13))
            cm = self.cm_metric.compute().cpu().detach().numpy()
            self.cm_metric.reset()
            ax = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
            # labels, title and ticks
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            ax.set_title('Confusion Matrix')
            ax.xaxis.set_ticklabels(self.classes, rotation=90)
            ax.yaxis.set_ticklabels(self.classes, rotation=0)
            plt.tight_layout()
            self.logger.experiment.add_figure("test/cm", fig, global_step=self.global_step)
        
        test_true = torch.cat([x['test_true'] for x in outputs])
        test_out = torch.cat([x['test_out'] for x in outputs])
