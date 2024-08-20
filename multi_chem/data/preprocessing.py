from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch

class graph_data:
	def __init__(self, smiles, label=None):
		self.smiles = smiles
		self.mol = Chem.MolFromSmiles(self.smiles)
		
		self.label = label

	def get_graph_feature(self):
		self.get_node_feature()
		self.get_edge_feature()
		self.get_edge_index()
		self.get_label()
		return [self.node_feature, self.edge_feature, self.edge_index, self.label]

	def one_hot_encoding(self, x, check_list):
		return list(map(lambda s: float(x == s), check_list))

	def get_node_feature(self):
		self.node_feature = []
		for idx, atom in enumerate(self.mol.GetAtoms()):
			assert idx == atom.GetIdx(), "In the function 'get_node_feature' in the class 'graph preprocessing', the index is wrong."
			self.node_feature.append(self.one_hot_encoding(int(atom.GetAtomicNum()), list(range(1,101))) +\
								self.one_hot_encoding(int(atom.GetTotalDegree()), list(range(0,6))) +\
								self.one_hot_encoding(int(atom.GetFormalCharge()), list(range(-2,3))) +\
								self.one_hot_encoding(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER']) +\
								self.one_hot_encoding(int(atom.GetTotalNumHs()), list(range(0,5))) +\
								self.one_hot_encoding(str(atom.GetHybridization()), ['SP','SP2','SP3','SP3D','SP3D2']) +\
								[float(atom.GetIsAromatic())] +\
								[float(atom.GetMass())*0.01])
		self.node_feature = torch.tensor(self.node_feature, dtype=torch.float)

	def get_edge_feature(self):
		self.edge_feature = []
		for bond in self.mol.GetBonds():
			edge_feat = self.one_hot_encoding(str(bond.GetBondType()), ['SINGLE','DOUBLE','TRIPLE','AROMATIC']) +\
						[float(bond.GetIsConjugated())] +\
						[float(bond.IsInRing())] +\
						self.one_hot_encoding(str(bond.GetStereo()), ['STEREONONE','STEREOANY','STEREOE','STEREOZ','STEREOCIS','STEREOTRANS'])
			self.edge_feature.append(edge_feat) #begin to end
			self.edge_feature.append(edge_feat) #end to begin
		self.edge_feature = torch.tensor(self.edge_feature, dtype=torch.float)

	def get_edge_index(self):
		self.edge_index = [[], []]
		for bond in self.mol.GetBonds():
			begin = bond.GetBeginAtomIdx()
			end = bond.GetEndAtomIdx()
			self.edge_index[0] += [begin, end] #start  begin end
			self.edge_index[1] += [end, begin] #finish end   begin
		self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)

	def get_label(self):
		if self.label is not None:
			self.label = torch.tensor(self.label, dtype=torch.float)
		else:
			pass
