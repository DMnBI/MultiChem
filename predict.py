if __name__ == '__main__':
	import os
	import sys

	data_idx = int(sys.argv[1])

	data_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox']
	batch_list = [64, 256, 256, 64, 64]
	label_list = [1, 12, 617, 27, 2]

	data = data_list[data_idx]
	batch = batch_list[data_idx]
	label = label_list[data_idx]

	run_cmd = 'python run.py \
			  --train_file /home/project/code_2024/MultiChem/example_data/mpg_data/{0}/fold_{1}/train.csv \
			  --val_file /home/project/code_2024/MultiChem/example_data/mpg_data/{0}/fold_{1}/valid.csv \
			  --test_file /home/project/code_2024/MultiChem/example_data/mpg_data/{0}/fold_{1}/test.csv \
			  --log_dir /home/project/code_2024/MultiChem/Log/{0}/fold_{1} \
			  --batch_size {2} --label_size {3} \
			  --predict'

	os.system(run_cmd.format(data, 0, batch, label))
	os.system(run_cmd.format(data, 1, batch, label))
	os.system(run_cmd.format(data, 2, batch, label))
