from torch.utils.data import Dataset

from .preprocessing import graph_data

from torch_geometric.data import Data

import torch
from torch.nn.functional import pad

class graph_dataset(Dataset):
	def __init__(self, inputs, labels=None):
		super().__init__()
	
		self.inputs = inputs
		self.labels = labels
		if self.labels is None:
			self.labels = [None for _ in range(len(self.inputs))]

	def __len__(self):
		return len(self.inputs)

	def __getitem__(self, idx):
		#graph
		data = graph_data(self.inputs[idx], self.labels[idx])
		node, edge, index, label = data.get_graph_feature()

		graph = Data(x=node, edge_index=index, y=label)
		
		#line graph
		node = torch.concat((node[index[0]], edge), dim=-1)
		if node.size(1) != 139:
			node = pad(node, (0,12), "constant", 0)

		_index = index.transpose(1,0)
		re_index = [[], []]
		for idx, (src, dst) in enumerate(_index):
			in_edge = ((_index[:,1] == src) & (_index[:,0] != dst)).nonzero(as_tuple=True)[0]
			for edge in in_edge:
				re_index[0].append(edge.item())
				re_index[1].append(idx)
		index = torch.tensor(re_index, dtype=torch.long)

		line_graph = Data(x=node, edge_index=index, y=label)
		return [graph, line_graph]
