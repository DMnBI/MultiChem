import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import torch
from torch.optim import NAdam

from pytorch_lightning import LightningModule

class graph_classifier(LightningModule):
	def __init__(self, model, learning_rate=1e-3, decay=0, **kwargs):
		super().__init__()
		self.model = model

		self.lr = learning_rate
		self.decay = decay

		self.results_roc = []
		self.results_prc = []

	def forward(self, batch):
		z, z1, z2 = self.model(batch[0], batch[1])

		loss = self.model.loss_function(z, batch[0].y)

		if z1 is not None and z2 is not None:
			loss1 = self.model.loss_function(z1, batch[0].y)
			loss2 = self.model.loss_function(z2, batch[0].y)
			loss = (loss + loss1 + loss2) / 3
			
		return loss, z, batch[0].y

	def configure_optimizers(self):
		optimizer = NAdam(self.parameters(), lr=self.lr, weight_decay=self.decay)
		return optimizer
		
	def training_step(self, batch, batch_idx):
		loss, pred, ans = self(batch)
		self.log("train_loss", loss, batch_size=ans.size(0), prog_bar=True, on_step=False, on_epoch=True)
		return {'loss':loss, 'predict':pred, 'answer':ans}

	def validation_step(self, batch, batch_idx):
		loss, pred, ans = self(batch)
		self.log("val_loss", loss, batch_size=ans.size(0), prog_bar=True, on_step=False, on_epoch=True)
		return {'loss':loss, 'predict':pred, 'answer':ans}

	def test_step(self, batch, batch_idx):
		loss, pred, ans = self(batch)
		self.log("test_loss", loss, batch_size=ans.size(0), prog_bar=True, on_step=False, on_epoch=True)
		return {'loss':loss, 'predict':pred, 'answer':ans}

	def predict_step(self, batch, batch_idx):
		loss, pred, ans = self(batch)
		return {'loss':loss, 'predict':pred, 'answer':ans}

	def test_epoch_end(self, outputs):
		pred, answ = self.process_to_auc(outputs)
		roc, prc = self.calculate_auc_scores(pred, answ)
		loss = torch.stack([out['loss'] for out in outputs])

		self.log(f'loss', torch.mean(loss).item(), prog_bar=True, on_step=False, on_epoch=True)
		self.log(f'roc', np.nanmean(roc), prog_bar=True, on_step=False, on_epoch=True)
		self.log(f'prc', np.nanmean(prc), prog_bar=True, on_step=False, on_epoch=True)

		self.results_roc.append(roc)
		self.results_prc.append(prc)

	def process_to_auc(self, outputs):
		pred = []
		answ = []
		for out in outputs:
			pred.append(out['predict'])
			answ.append(out['answer'])
		pred = torch.concat(pred, dim=0)
		pred = torch.sigmoid(pred)
		answ = torch.concat(answ, dim=0)
		return pred, answ

	def calculate_auc_scores(self, pred, answ):
		roc_scores = []
		prc_scores = []
		for idx in range(pred.shape[1]):
			p = pred[:, idx]
			a = answ[:, idx]
			p = p[a != -1]
			a = a[a != -1]
			p = p.detach().cpu().numpy()
			a = a.detach().cpu().numpy()
			try:
				roc_scores.append(roc_auc_score(a, p))
			except:
				roc_scores.append(None)
			try:
				prc_scores.append(average_precision_score(a, p))
			except:
				prc_scores.append(None)

		roc_scores = np.array(roc_scores, dtype=float)
		prc_scores = np.array(prc_scores, dtype=float)
		return roc_scores, prc_scores
