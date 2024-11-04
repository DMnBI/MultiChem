import numpy as np
from sklearn.metrics import mean_squared_error

import torch
from torch.optim import NAdam

from pytorch_lightning import LightningModule

class graph_regressor(LightningModule):
	def __init__(self, model, learning_rate=1e-3, decay=0, **kwargs):
		super().__init__()
		self.model = model

		self.lr = learning_rate
		self.decay = decay

		self.results_rmse = []

	def forward(self, batch):
		z, z1, z2 = self.model(batch[0], batch[1])

		loss = self.model.loss_function_reg(z, batch[0].y)

		if z1 is not None and z2 is not None:
			loss1 = self.model.loss_function_reg(z1, batch[0].y)
			loss2 = self.model.loss_function_reg(z2, batch[0].y)
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
		pred = []
		answ = []
		for out in outputs:
			pred.append(out['predict'])
			answ.append(out['answer'])
		pred = torch.concat(pred, dim=0)
		answ = torch.concat(answ, dim=0)

		pred = pred[:, 0]
		answ = answ[:, 0]

		pred = pred.detach().cpu().numpy()
		answ = answ.detach().cpu().numpy()

		loss = torch.stack([out['loss'] for out in outputs])
		rmse = np.sqrt(mean_squared_error(answ, pred))

		self.log(f'loss', torch.mean(loss).item(), prog_bar=True, on_step=False, on_epoch=True)
		self.log(f'rmse', rmse, prog_bar=True, on_step=False, on_epoch=True)

		self.results_rmse.append(rmse)
