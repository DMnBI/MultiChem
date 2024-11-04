if __name__ == '__main__':
	import os
	import sys

	data_idx = int(sys.argv[1])
	which_dataset = str(sys.argv[2])

                 #0      #1       #2         #3       #4         #5      #6          #7      #8	
	data_list = ['bbbp', 'tox21', 'toxcast', 'sider', 'clintox', 'bace', 'freesolv', 'esol', 'lipophilicity']
	batch_list = [64, 256, 256, 64, 64, 32, 16, 32, 128]
	label_list = [1, 12, 617, 27, 2, 1, 1, 1, 1]
	tasktype_list = [0, 0, 0, 0, 0, 0, 1, 1, 1]

	data = data_list[data_idx]
	batch = batch_list[data_idx]
	label = label_list[data_idx]
	tasktype = tasktype_list[data_idx]
	
	if which_dataset == 'a':
		print(f'Learning on {data} in GROVER dataset')

		run_cmd = 'python run.py \
				  --train_file /home/project/code_2024/MultiChem/example_data/grover_data/{0}/fold_{1}/train.csv \
				  --val_file /home/project/code_2024/MultiChem/example_data/grover_data/{0}/fold_{1}/valid.csv \
				  --test_file /home/project/code_2024/MultiChem/example_data/grover_data/{0}/fold_{1}/test.csv \
				  --log_dir /home/project/code_2024/MultiChem/Log_grover/{0}/fold_{1} \
				  --batch_size {2} --label_size {3} \
				  --learning \
				  --task_type {4}'
	elif which_dataset == 'b':
		print(f'Learning on {data} in MPG dataset')

		run_cmd = 'python run.py \
				  --train_file /home/project/code_2024/MultiChem/example_data/mpg_data/{0}/fold_{1}/train.csv \
				  --val_file /home/project/code_2024/MultiChem/example_data/mpg_data/{0}/fold_{1}/valid.csv \
				  --test_file /home/project/code_2024/MultiChem/example_data/mpg_data/{0}/fold_{1}/test.csv \
				  --log_dir /home/project/code_2024/MultiChem/Log/{0}/fold_{1} \
				  --batch_size {2} --label_size {3} \
				  --learning \
				  --task_type {4}'
	else:
		exit()

	os.system(run_cmd.format(data, 0, batch, label, tasktype))
	os.system(run_cmd.format(data, 1, batch, label, tasktype))
	os.system(run_cmd.format(data, 2, batch, label, tasktype))
