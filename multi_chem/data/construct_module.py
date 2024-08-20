from pytorch_lightning import LightningDataModule

from .load_data import load_file, rebuild
from .build_set import graph_dataset

from torch.utils.data import DataLoader

from torch_geometric.data import Batch

class graph_datamodule(LightningDataModule):
	def __init__(self, train_file=None, val_file=None, test_file=None, batch_size=128, num_workers=4, e_seed=-1, **kwargs):
		super().__init__()

		if train_file is not None and val_file is not None:
			self.train_set, self.val_set = self.get_rebuild_dataset(train_file, val_file, e_seed)

			self.test_set = self.get_dataset(test_file)
		else:
			self.train_set = self.get_dataset(train_file)
			self.val_set = self.get_dataset(val_file)
			self.test_set = self.get_dataset(test_file)

		self.batch_size = batch_size
		self.num_workers = num_workers

	def get_dataset(self, file_path):
		if file_path is not None:
			loader = load_file()
			data = loader.csv_to_df(file_path)
			data = loader.df_to_data(data)
			return graph_dataset(**data)
		else:
			return None

	def get_rebuild_dataset(self, train_path, val_path, e_seed):
		loader = load_file()
		train = loader.csv_to_df(train_path)
		val = loader.csv_to_df(val_path)

		train, val = rebuild().rebuild_df(train, val, e_seed)

		train = loader.df_to_data(train)
		val = loader.df_to_data(val)
		return [graph_dataset(**train), graph_dataset(**val)]

	def collate_fn(self, batch):
		graph_list = []
		line_graph_list = []
		for graph, line_graph in batch:
			graph_list.append(graph)
			line_graph_list.append(line_graph)

		cur_batch_size = len(graph_list)

		batch_graph = Batch.from_data_list(graph_list)
		if batch_graph.y is not None:
			batch_graph.y = batch_graph.y.view(cur_batch_size, -1)
		else:
			None

		batch_line_graph = Batch.from_data_list(line_graph_list)
		if batch_line_graph.y is not None:
			batch_line_graph.y = batch_line_graph.y.view(cur_batch_size, -1)
		else:
			None

		return batch_graph, batch_line_graph

	def train_dataloader(self):
		if self.train_set is not None:
			return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, 
					persistent_workers=True, pin_memory=True, shuffle=True)
		else:
			raise FileNotFoundError

	def val_dataloader(self):
		if self.val_set is not None:
			return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, 
					persistent_workers=True, pin_memory=True)
		else:
			raise FileNotFoundError

	def test_dataloader(self):
		if self.test_set is not None:
			return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn, 
					persistent_workers=True, pin_memory=True)
		else:
			raise FileNotFoundError
