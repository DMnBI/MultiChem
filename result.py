if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--log_dir', default='/home/project/code_2024/MultiChem/default_log')
	parser.add_argument('--data', default='tox21')
	parser.add_argument('--measure', default='roc')

	args = parser.parse_args()

	import os
	import pandas as pd

	def get_result(log_dir, data):
		def get_dataframe(log_dir, data, i):
			df = pd.read_csv(os.path.join(log_dir, data, f'fold_{i}', 'ensemble_result.csv'))

			df = df.rename(index={0:'avg'})
			df = df.rename(columns={'test_roc':f'roc{i}'})
			df = df.rename(columns={'test_prc':f'prc{i}'})

			roc_df = df[f'roc{i}']
			prc_df = df[f'prc{i}']
			return roc_df, prc_df

		roc_0, prc_0 = get_dataframe(log_dir, data, 0)
		roc_1, prc_1 = get_dataframe(log_dir, data, 1)
		roc_2, prc_2 = get_dataframe(log_dir, data, 2)

		def concat_result(df_0, df_1, df_2):
			df = pd.concat([df_0, df_1, df_2], axis=1)
			return df
		
		roc = concat_result(roc_0, roc_1, roc_2)
		prc = concat_result(prc_0, prc_1, prc_2)

		def mean_and_std(df):
			df['mean'] = df.mean(axis=1)
			df['std'] = df.std(axis=1)
			return df

		roc = mean_and_std(roc)
		prc = mean_and_std(prc)
		
		return roc, prc

	result_roc, result_prc = get_result(args.log_dir, args.data)

	pd.set_option('display.float_format', lambda x: f'{x:.3f}')

	if args.measure == 'roc':
		print(result_roc)
	elif args.measure == 'prc':
		print(result_prc)
	else:
		None
