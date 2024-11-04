import random
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

def generate_scaffold(mol, include_chirality=False):
	mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
	scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
	return scaffold

def scaffold_to_smiles(mols, use_indices=False):
	scaffolds = defaultdict(set)
	for i, mol in enumerate(mols):
		scaffold = generate_scaffold(mol)
		if use_indices:
			scaffolds[scaffold].add(i)
		else:
			scaffolds[scaffold].add(mol)
	return scaffolds

def scaffold_split(data, sizes=(0.8, 0.1, 0.1), seed=0):
	assert sum(sizes) == 1

	train_size, val_size, test_size = sizes[0] * len(data), sizes[1] * len(data), sizes[2] * len(data)
	train, val, test = [], [], []
	train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

	scaffold_to_indices = scaffold_to_smiles(data['smiles'], use_indices=True)

	index_sets = list(scaffold_to_indices.values())
	big_index_sets = []
	small_index_sets = []
	for index_set in index_sets:
		if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
			big_index_sets.append(index_set)
		else:
			small_index_sets.append(index_set)

	random.seed(seed)
	random.shuffle(big_index_sets)
	random.shuffle(small_index_sets)
	index_sets = big_index_sets + small_index_sets

	for index_set in index_sets:
		if len(train) + len(index_set) <= train_size:
			train += index_set
			train_scaffold_count += 1
		elif len(val) + len(index_set) <= val_size:
			val += index_set
			val_scaffold_count += 1
		else:
			test += index_set
			test_scaffold_count += 1

	train = data.iloc[train]
	val = data.iloc[val]
	test = data.iloc[test]
	return train, val, test

import os
import sys
import pandas as pd

if __name__ == '__main__':

	data_name = sys.argv[1]

	def make_split_data(data_name, seed):
		read_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f'origin/{data_name}.csv')
		df = pd.read_csv(read_path)

		train, val, test = scaffold_split(df, seed=seed)

		write_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), f'{data_name}/fold_{seed}')
		os.makedirs(write_path)

		train_path = os.path.join(write_path, 'train.csv')
		train.to_csv(train_path, sep=',', index=False)
		val_path = os.path.join(write_path, 'valid.csv')
		val.to_csv(val_path, sep=',', index=False)
		test_path = os.path.join(write_path, 'test.csv')
		test.to_csv(test_path, sep=',', index=False)

	for i in range(30):
		make_split_data(data_name, i)
