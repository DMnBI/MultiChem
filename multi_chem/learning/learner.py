import os
import random
import numpy as np
import pandas

import torch

from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning import Trainer

from ..data.construct_module import graph_datamodule, graph_datamodule_reg
from ..module.gnn import MultiChem
from ..plmodule.classifier import graph_classifier
from ..plmodule.regressor import graph_regressor, mean_squared_error

class learn_MultiChem:
	def __init__(self, log_dir='/home/MultiChem/log', patience=50, epoch=5000, gpus=[3], 
			learning=False, predict=False, **kwargs):

		self.log_dir = log_dir
		self.patience = patience
		self.epoch = epoch
		self.gpus = gpus

		self.learning = learning
		self.predict = predict

		self.args_dict = kwargs

	def run(self):
		if self.learning and self.predict:
			raise ValueError
		elif self.learning:
			self.get_learn()
		elif self.predict:
			self.get_predict()
		else:
			raise ValueError

	def get_learn(self):
		for e_seed in [-1, 0, 1, 2, 3, 4]:
			self.args_dict['e_seed'] = e_seed

			os.environ["PYTHONHASHSEED"] = str(1)
			random.seed(1)
			np.random.seed(1)
			torch.manual_seed(1)
			torch.cuda.manual_seed(1)
			torch.cuda.manual_seed_all(1)

			data = graph_datamodule(**self.args_dict)
		
			model = MultiChem(**self.args_dict)
			model = graph_classifier(model, **self.args_dict)

			summary = RichModelSummary(max_depth=2)
			progressbar = RichProgressBar()
			
			checkpoint = ModelCheckpoint(dirpath=os.path.join(self.log_dir, 'checkpoint'), monitor='val_loss', mode='min', filename=f'single_{e_seed+1}')
			earlystop = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min')

			logger_csv = CSVLogger(save_dir=self.log_dir, name='log_csv')
			tensorboard = TensorBoardLogger(save_dir=self.log_dir, name='log_tensor_board')

			self.trainer = Trainer(max_epochs=self.epoch, accelerator="gpu", default_root_dir=self.log_dir, 
					callbacks=[summary, progressbar, checkpoint, earlystop], logger=[logger_csv, tensorboard])

			self.trainer.fit(model, datamodule=data)

			self.trainer.test(model, dataloaders=data.train_dataloader(), ckpt_path='best', verbose=False)
			self.trainer.test(model, dataloaders=data.val_dataloader(), ckpt_path='best', verbose=False)
			self.trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path='best', verbose=False)

			print(np.nanmean(np.array(model.results_roc).reshape(3, -1), 1))

	def get_predict(self):
		preds = []
		answs = []
		for e_seed in [-1, 0, 1, 2, 3, 4]:
			self.args_dict['e_seed'] = e_seed

			data = graph_datamodule(**self.args_dict)
		
			model = MultiChem(**self.args_dict)
			model = graph_classifier(model, **self.args_dict)

			summary = RichModelSummary(max_depth=2)
			progressbar = RichProgressBar()

			trainer = Trainer(max_epochs=self.epoch, accelerator="gpu", default_root_dir=self.log_dir, 
					callbacks=[summary, progressbar], logger=[])

			ckpt = os.path.join(self.log_dir, 'checkpoint', f'single_{e_seed+1}.ckpt')
			outputs = trainer.predict(model, dataloaders=data.test_dataloader(), ckpt_path=ckpt)

			pred, answ = model.process_to_auc(outputs)
			preds.append(pred)
			answs.append(answ)

		preds = torch.mean(torch.stack(preds, dim=1), dim=1)
		answs = torch.mean(torch.stack(answs, dim=1), dim=1)

		roc, prc = model.calculate_auc_scores(preds, answs)

		columns = ['test_roc', 'test_prc']
		results = [roc, prc]
		means = np.nanmean(results, axis=1)

		df = pandas.DataFrame()
		for column, result, mean in zip(columns, results, means):
			df[column] = np.insert(result, 0, mean)
		df.rename(index={0:'avg'}, inplace=True)
			
		df.to_csv(os.path.join(self.log_dir, 'ensemble_result.csv'))

class learn_MultiChem_reg:
	def __init__(self, log_dir='/home/MultiChem/log', patience=50, epoch=5000, gpus=[3], 
			learning=False, predict=False, **kwargs):

		self.log_dir = log_dir
		self.patience = patience
		self.epoch = epoch
		self.gpus = gpus

		self.learning = learning
		self.predict = predict

		self.args_dict = kwargs

	def run(self):
		if self.learning and self.predict:
			raise ValueError
		elif self.learning:
			self.get_learn()
		elif self.predict:
			self.get_predict()
		else:
			raise ValueError

	def get_learn(self):
		for e_seed in [-1, 0, 1, 2, 3, 4]:
			self.args_dict['e_seed'] = e_seed

			os.environ["PYTHONHASHSEED"] = str(1)
			random.seed(1)
			np.random.seed(1)
			torch.manual_seed(1)
			torch.cuda.manual_seed(1)
			torch.cuda.manual_seed_all(1)

			data = graph_datamodule_reg(**self.args_dict)
		
			model = MultiChem(**self.args_dict)
			model = graph_regressor(model, **self.args_dict)

			summary = RichModelSummary(max_depth=2)
			progressbar = RichProgressBar()
			
			checkpoint = ModelCheckpoint(dirpath=os.path.join(self.log_dir, 'checkpoint'), monitor='val_loss', mode='min', filename=f'single_{e_seed+1}')
			earlystop = EarlyStopping(monitor='val_loss', patience=self.patience, mode='min')

			logger_csv = CSVLogger(save_dir=self.log_dir, name='log_csv')
			tensorboard = TensorBoardLogger(save_dir=self.log_dir, name='log_tensor_board')

			self.trainer = Trainer(max_epochs=self.epoch, accelerator="gpu", default_root_dir=self.log_dir, 
					callbacks=[summary, progressbar, checkpoint, earlystop], logger=[logger_csv, tensorboard])

			self.trainer.fit(model, datamodule=data)

			self.trainer.test(model, dataloaders=data.train_dataloader(), ckpt_path='best', verbose=False)
			self.trainer.test(model, dataloaders=data.val_dataloader(), ckpt_path='best', verbose=False)
			self.trainer.test(model, dataloaders=data.test_dataloader(), ckpt_path='best', verbose=False)

			print(np.mean(np.array(model.results_rmse).reshape(3, -1), 1))

	def get_predict(self):
		preds = []
		answs = []

		for e_seed in [-1, 0, 1, 2, 3, 4]:
			self.args_dict['e_seed'] = e_seed

			data = graph_datamodule_reg(**self.args_dict)
		
			model = MultiChem(**self.args_dict)
			model = graph_regressor(model, **self.args_dict)

			summary = RichModelSummary(max_depth=2)
			progressbar = RichProgressBar()

			trainer = Trainer(max_epochs=self.epoch, accelerator="gpu", default_root_dir=self.log_dir, 
					callbacks=[summary, progressbar], logger=[])

			ckpt = os.path.join(self.log_dir, 'checkpoint', f'single_{e_seed+1}.ckpt')
			outputs = trainer.predict(model, dataloaders=data.test_dataloader(), ckpt_path=ckpt)

			pred = []
			answ = []
			for out in outputs:
				pred.append(out['predict'])
				answ.append(out['answer'])
			pred = torch.concat(pred, dim=0)
			answ = torch.concat(answ, dim=0)

			preds.append(pred)
			answs.append(answ)

		preds = torch.mean(torch.stack(preds, dim=1), dim=1)
		answs = torch.mean(torch.stack(answs, dim=1), dim=1)

		pred = preds[:, 0]
		answ = answs[:, 0]

		pred = pred.detach().cpu().numpy()
		answ = answ.detach().cpu().numpy()

		rmse = np.sqrt(mean_squared_error(answ, pred))

		rmse = rmse.reshape(1, -1)
		columns = ['test_rmse']
		results = [rmse]
		means = np.nanmean(results, axis=1)

		df = pandas.DataFrame()
		for column, result, mean in zip(columns, results, means):
			df[column] = np.insert(result, 0, mean)
		df.rename(index={0:'avg'}, inplace=True)
			
		df.to_csv(os.path.join(self.log_dir, 'ensemble_result.csv'))
