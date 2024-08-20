import torch
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import Module
from torch.nn import ModuleList

import torch.nn.functional as F

from .layer import gnn_block
from .layer import edge_graph_pool
from .layer import merge
from .layer import graph_to_seq
from .layer import position_encoding
from .layer import attention_block
from .layer import graph_pool
from .layer import ffnn_block

class MultiChem(Module):
	def __init__(self, node_size=127, layer_size=128, layer_depth=3, 
				label_size=None, dropout=0.3, edge_size=None, heads=4, **kwargs):
		super().__init__()

		self.layer_size = layer_size
		self.layer_depth = layer_depth

		self.atom_init_layer = Linear(node_size, layer_size)
		self.bond_init_layer = Linear(node_size + edge_size, layer_size)

		self.atom_blocks = ModuleList()
		self.bond_blocks = ModuleList()
		for i in range(0, layer_depth):
			self.atom_blocks.append(gnn_block(layer_size, layer_size, heads, dropout, layer_size))
			self.bond_blocks.append(gnn_block(layer_size, layer_size, heads, dropout, layer_size))

		self.pooling_edge = edge_graph_pool(layer_size, dropout)
		self.merge_layer1 = merge(layer_size, dropout)

		self.batch_layer = graph_to_seq(heads)
		self.position = position_encoding(layer_size)
		self.norm = LayerNorm(layer_size)

		self.attention = attention_block(layer_size, layer_size, heads, dropout)
		self.merge_layer2 = merge(layer_size, dropout)

		self.pooling_graph = graph_pool(layer_size, dropout)

		#self.ffnn = ffnn_block(layer_size, label_size, dropout)
		self.ffnn = Linear(layer_size, label_size)

		self.pooling_z1 = graph_pool(layer_size, dropout)
		self.pooling_z2 = graph_pool(layer_size, dropout)

		self.ffnn_z1 = Linear(layer_size, label_size)
		self.ffnn_z2 = Linear(layer_size, label_size)

	def loss_function(self, output, target):
		loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
		loss[target == -1] = torch.nan
		loss = loss.nanmean()
		loss = loss.nan_to_num()
		return loss

	def forward(self, batch, lin_batch):
		node_x = self.atom_init_layer(batch.x)
		edge_x = self.bond_init_layer(lin_batch.x)

		for i in range(0, self.layer_depth):
			node_for_edge = node_x.index_select(0, batch.edge_index[0])
			node_for_edge = node_for_edge.index_select(0, lin_batch.edge_index[0])

			node_x = self.atom_blocks[i](node_x, batch.edge_index, edge_x)
			edge_x = self.bond_blocks[i](edge_x, lin_batch.edge_index, node_for_edge)

		edge_x = self.pooling_edge(edge_x, batch.edge_index, batch.batch)
		graph_z = self.merge_layer1(node_x, edge_x)

		x, masks, att_masks = self.batch_layer(batch.batch, graph_z, self.layer_size)

		pos = self.position(x)
		x = self.norm(x + pos)

		seq_z = self.attention(x, att_masks)
		seq_z = seq_z[masks]

		z = self.merge_layer2(graph_z, seq_z)
		z = self.pooling_graph(z, batch.batch)

		z = self.ffnn(z)

		z1 = self.pooling_z1(node_x, batch.batch)
		z2 = self.pooling_z2(edge_x, batch.batch)

		z1 = self.ffnn_z1(z1)
		z2 = self.ffnn_z2(z2)

		return z, z1, z2
