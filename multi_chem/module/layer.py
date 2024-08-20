import math
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import GELU
from torch.nn import Dropout
from torch.nn import MultiheadAttention
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax

class gnn_block(Module):
	def __init__(self, input_size, output_size, heads, dropout, edge_dim):
		super().__init__()
		self.conv = TrimConv(in_channels=input_size, out_channels=output_size//heads, heads=heads, dropout=dropout, edge_dim=edge_dim, concat=True)
		self.norm1 = LayerNorm(output_size)

		self.lin1 = Linear(output_size, output_size)
		self.act = GELU()
		self.drop = Dropout(dropout)
		self.lin2 = Linear(output_size, output_size)
		self.norm2 = LayerNorm(output_size)
	
	def forward(self, x, edge_index, edge_attr=None):
		h = self.conv(x, edge_index, edge_attr)
		h = self.norm1(x + h)

		z = self.lin1(h)
		z = self.act(z)
		z = self.drop(z)
		z = self.lin2(z)
		z = self.norm2(h + z)
		return z

class TrimConv(GATConv):
	def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor, 
			edge_attr: OptTensor, index: Tensor, ptr: OptTensor, 
			size_i: Optional[int]) -> Tensor:
		alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
		
		if edge_attr is not None and self.lin_edge is not None:
			if edge_attr.dim() == 1:
				edge_attr = edge_attr.view(-1, 1)
			edge_attr = self.lin_edge(edge_attr)
			self.edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
			alpha_edge = (self.edge_attr * self.att_edge).sum(dim=-1)
			alpha = alpha + alpha_edge

		alpha = F.leaky_relu(alpha, self.negative_slope)
		alpha = softmax(alpha, index, ptr, size_i)
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)
		return alpha
		
	def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
		return alpha.unsqueeze(-1) * x_j * self.edge_attr

class edge_graph_pool(Module):
	def __init__(self, layer_size, dropout):
		super().__init__()

		self.aggr = MeanAggregation()
		self.norm1 = LayerNorm(layer_size)
		self.lin1 = Linear(layer_size, layer_size)
		self.act = GELU()
		self.drop = Dropout(dropout)
		self.lin2 = Linear(layer_size, layer_size)
		self.norm2 = LayerNorm(layer_size)

	def forward(self, edge_graph, edge_index, graph_index):
		h = self.aggr(edge_graph, edge_index[1], dim_size=len(graph_index))
		h = self.norm1(h)
		z = self.lin1(h)
		z = self.act(z)
		z = self.drop(z)
		z = self.lin2(z)
		z = self.norm2(h + z)
		return z

class merge(Module):
	def __init__(self, output_size, dropout):
		super().__init__()
		self.lin1 = Linear(output_size, output_size)
		self.lin2 = Linear(output_size, output_size)
		self.norm1 = LayerNorm(output_size)

		self.lin3 = Linear(output_size, output_size)
		self.act = GELU()
		self.drop = Dropout(dropout)
		self.lin4 = Linear(output_size, output_size)
		self.norm2 = LayerNorm(output_size)
	
	def forward(self, a, b):
		x = self.lin1(a)
		y = self.lin2(b)
		h = self.norm1(x + y)

		z = self.lin3(h)
		z = self.act(z)
		z = self.drop(z)
		z = self.lin4(z)
		z = self.norm2(h + z)
		return z

class graph_to_seq:
	def __init__(self, heads):
		self.heads = heads

	def __call__(self, which_batch, z, pad_size):
		batch_size = len(which_batch.unique())

		batch_stack = []
		for b_idx in range(batch_size):
			mask = (which_batch == b_idx)
			batch_stack.append(z[mask])
		batch = pad_sequence(batch_stack, batch_first=True)
		batch = F.pad(batch, (0, pad_size-batch.size(-1)))

		max_len = batch[0].size()[0]

		mask_stack = []
		for idx in range(batch_size):
			cutoff = batch_stack[idx].size()[0]
			mask = torch.zeros(max_len, dtype=torch.bool, device=z.device)
			mask[:cutoff] = True
			mask_stack.append(mask)
		masks = torch.concat(mask_stack, dim=0)

		att_masks = masks.reshape(batch_size, -1)
		att_masks = att_masks.unsqueeze(dim=1)
		att_masks = att_masks.type(torch.float32)
		att_masks = torch.matmul(att_masks.transpose(2,1), att_masks)
		temp = torch.eye(att_masks.size()[1], device=z.device).unsqueeze(0)
		att_masks = att_masks + temp
		att_masks[att_masks > 1] = 1
		att_masks = att_masks - 1
		att_masks = att_masks * -1
		att_masks = att_masks.type(torch.bool)
		att_masks = torch.repeat_interleave(att_masks, self.heads, dim=0)

		masks = masks.reshape(batch_size, -1)
		return batch, masks, att_masks

class position_encoding(Module):
	def __init__(self, even_length, max_len=512):
		super().__init__()

		pos_tensor = torch.zeros(max_len, even_length).float()
		pos_tensor.require_grad = False

		position = torch.arange(0, max_len).float().unsqueeze(1)
		div_term = (torch.arange(0, even_length, 2).float() * -(math.log(10000.0) / even_length)).exp()

		pos_tensor[:, 0::2] = torch.sin(position * div_term)
		pos_tensor[:, 1::2] = torch.cos(position * div_term)

		pos_tensor = pos_tensor.unsqueeze(0)
		self.register_buffer('pos_tensor', pos_tensor)

	def forward(self, x):
		return self.pos_tensor[:, :x.size(1)]

class attention_block(Module):
	def __init__(self, input_size, output_size, heads, dropout):
		super().__init__()
		self.query_layer = Linear(input_size, output_size)
		self.key_layer = Linear(input_size, output_size)
		self.value_layer = Linear(input_size, output_size)
		self.attention_layer = MultiheadAttention(output_size, heads, dropout, batch_first=True)
		self.norm_layer1 = LayerNorm(output_size)

		self.dense_layer1 = Linear(output_size, output_size)
		self.act_layer = GELU()
		self.drop_layer = Dropout(dropout)
		self.dense_layer2 = Linear(output_size, output_size)
		self.norm_layer2 = LayerNorm(output_size)
	
	def forward(self, x, attn_masks=None):
		q = self.query_layer(x)
		k = self.key_layer(x)
		v = self.value_layer(x)
		if attn_masks is not None:
			h, att_weight = self.attention_layer(q, k, v, attn_mask=attn_masks)
		else:
			h, att_weight = self.attention_layer(q, k, v)
		h = self.norm_layer1(x + h)

		z = self.dense_layer1(h)
		z = self.act_layer(z)
		z = self.drop_layer(z)
		z = self.dense_layer2(z)
		z = self.norm_layer2(h + z)
		return z

class graph_pool(Module):
	def __init__(self, layer_size, dropout):
		super().__init__()

		self.aggr = global_mean_pool
		self.norm1 = LayerNorm(layer_size)
		self.lin1 = Linear(layer_size, layer_size)
		self.act = GELU()
		self.drop = Dropout(dropout)
		self.lin2 = Linear(layer_size, layer_size)
		self.norm2 = LayerNorm(layer_size)
	
	def forward(self, x, graph_index):
		h = self.aggr(x, graph_index)
		h = self.norm1(h)
		z = self.lin1(h)
		z = self.act(z)
		z = self.drop(z)
		z = self.lin2(z)
		z = self.norm2(h + z)
		return z

class ffnn_block(Module):
	def __init__(self, hidden_size, output_size, dropout):
		super().__init__()

		self.lin1 = Linear(hidden_size, hidden_size)
		self.act = GELU()
		self.drop = Dropout(dropout)
		self.lin2 = Linear(hidden_size, output_size)

	def forward(self, x):
		h = self.lin1(x)
		h = self.act(h)
		h = self.drop(h)
		h = self.lin2(h)
		return h
