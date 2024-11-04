import pandas
import random
from deepchem.splits.splitters import _generate_scaffold

class load_file:
	def csv_to_df(self, file_path):
		data = pandas.read_csv(file_path, index_col=False).fillna(-1)

		col_names = data.columns

		if 'num' in col_names and 'name' in col_names: 
			data = data.drop(['num', 'name'], axis=1)
		elif 'mol_id' in col_names:
			data = data.drop(['mol_id'], axis=1)
		elif 'ID' in col_names and 'Unnamed: 0' in col_names:
			data = data.drop(['ID', 'Unnamed: 0'], axis=1)
			data = data.rename(columns={'SMILES':'smiles'})
		elif 'mol' in col_names and 'Class' in col_names:
			data = data.rename(columns={'mol':'smiles'})
			data = data[['smiles', 'Class']]
		else:
			pass

		return data

	def df_to_data(self, data):
		if len(data.columns) > 1:
			data = {'inputs':data['smiles'].to_numpy(), 'labels':data.loc[:, ~data.columns.isin(['smiles'])].to_numpy()}
		else:
			data = {'inputs':data['smiles'].to_numpy(), 'labels':None}
		return data

class rebuild:
	def rebuild_df(self, train, val, e_seed):
		if e_seed == -1:
			pass
		else:
			dataset = pandas.concat([train, val])
			dataset = dataset.reset_index(drop=True)

			scaffolds = {}
			data_len = len(dataset)

			for ind, smiles in enumerate(dataset['smiles']):
				scaffold = _generate_scaffold(smiles)
				if scaffold not in scaffolds:
					scaffolds[scaffold] = [ind]
				else:
					scaffolds[scaffold].append(ind)

			scaffold_sets = list(scaffolds.values())

			train_size = len(train)
			val_size = len(val)

			big_sets = []
			small_sets = []
			for scaffold in scaffold_sets:
				if len(scaffold) > val_size/2:
					big_sets.append(scaffold)
				else:
					small_sets.append(scaffold)

			random.seed(e_seed)
			random.shuffle(big_sets)
			random.shuffle(small_sets)

			scaffold_sets = big_sets + small_sets

			train_inds: List[int] = []
			valid_inds: List[int] = []

			for scaffold in scaffold_sets:
				if len(train_inds) + len(scaffold) <= train_size:
					train_inds += scaffold
				else:
					valid_inds += scaffold

			train = dataset.iloc[train_inds]
			train = train.reset_index(drop=True)
			val = dataset.iloc[valid_inds]
			val = val.reset_index(drop=True)

		return train, val

class load_file_reg:
	def csv_to_df(self, file_path):
		data = pandas.read_csv(file_path, index_col=False)

		col_names = data.columns

		if 'expt' in col_names:
			data = data[['smiles', 'expt']]
		elif 'measured log solubility in mols per litre' in col_names:
			data = data[['smiles', 'measured log solubility in mols per litre']]
		elif 'exp' in col_names:
			data = data[['smiles', 'exp']]
		else:
			pass

		return data

	def df_to_data(self, data):
		if len(data.columns) > 1:
			data = {'inputs':data['smiles'].to_numpy(), 'labels':data.loc[:, ~data.columns.isin(['smiles'])].to_numpy()}
		else:
			data = {'inputs':data['smiles'].to_numpy(), 'labels':None}
		return data
