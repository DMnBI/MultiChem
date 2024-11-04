import os
import pandas as pd

def mean_and_std(df):
	df['mean'] = df.mean(axis=1)
	df['std'] = df.std(axis=1)
	return df

def get_roc_dataframe(log_dir, data, i):
	ensemble = pd.read_csv(os.path.join(log_dir, data, f'fold_{i}', 'ensemble_result.csv')).iloc[:1]
	ensemble = ensemble[['test_roc']]
	ensemble = ensemble.rename(index={0:f'ensemble'})
	ensemble = ensemble.rename(columns={'test_roc':f'roc{i}'})

	single_list = []
	for j in [0,1,2,3,4,5]:
		df = pd.read_csv(os.path.join(log_dir, data, f'fold_{i}', 'log_csv', f'version_{j}', 'metrics.csv')).iloc[-1:]
		df = df[['roc']].reset_index(drop=True)
		df = df.rename(index={0:f'single{j}'})
		df = df.rename(columns={'roc':f'roc{i}'})
		single_list.append(df)

	df = pd.concat([ensemble] + single_list, axis=0)
	return df

def get_rmse_dataframe(log_dir, data, i):
	ensemble = pd.read_csv(os.path.join(log_dir, data, f'fold_{i}', 'ensemble_result.csv')).iloc[:1]
	ensemble = ensemble[['test_rmse']]
	ensemble = ensemble.rename(index={0:f'ensemble'})
	ensemble = ensemble.rename(columns={'test_rmse':f'rmse{i}'})

	single_list = []
	for j in [0,1,2,3,4,5]:
		df = pd.read_csv(os.path.join(log_dir, data, f'fold_{i}', 'log_csv', f'version_{j}', 'metrics.csv')).iloc[-1:]
		df = df[['rmse']].reset_index(drop=True)
		df = df.rename(index={0:f'single{j}'})
		df = df.rename(columns={'rmse':f'rmse{i}'})
		single_list.append(df)

	df = pd.concat([ensemble] + single_list, axis=0)
	return df

def get_roc_result(log_dir, data, n_seed):
	roc_list = []
	for i in range(n_seed):
		roc = get_roc_dataframe(log_dir, data, i)
		roc_list.append(roc)

	roc = pd.concat(roc_list, axis=1)
	roc = mean_and_std(roc)
	return roc

def get_rmse_result(log_dir, data, n_seed):
	rmse_list = []
	for i in range(n_seed):
		rmse = get_rmse_dataframe(log_dir, data, i)
		rmse_list.append(rmse)

	rmse = pd.concat(rmse_list, axis=1)
	rmse = mean_and_std(rmse)
	return rmse

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--log_dir', default='/home/project/code_2024/MultiChem/Log')
	parser.add_argument('--data', default='tox21')
	parser.add_argument('--measure', default='roc')
	parser.add_argument('--n_seed', default=3, type=int)

	args = parser.parse_args()

	print(args.log_dir)
	print(args.data)
	print(args.measure)
	print(args.n_seed)

	pd.set_option('display.float_format', lambda x: f'{x:.3f}')

	if args.measure == 'roc':
		result_roc = get_roc_result(args.log_dir, args.data, args.n_seed)
		print(result_roc)
		print(result_roc[['mean', 'std']])

		for i in range(len(result_roc)):
			_str = ' '.join(map(str, result_roc.iloc[i].to_list()))
#			print(_str)
	elif args.measure == 'rmse':
		result_rmse = get_rmse_result(args.log_dir, args.data, args.n_seed)
		print(result_rmse)
		print(result_rmse[['mean', 'std']])

		for i in range(len(result_rmse)):
			_str = ' '.join(map(str, result_rmse.iloc[i].to_list()))
#			print(_str)
	else:
		None
