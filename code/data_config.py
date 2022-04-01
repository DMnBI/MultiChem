import os

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        None

cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.abspath(os.path.join(cur_dir, os.pardir))

data_dir = os.path.join(par_dir, 'data')
create_dir(data_dir)

origin_dir = os.path.join(data_dir, 'origin_data')
create_dir(origin_dir)

split_dir = os.path.join(data_dir, 'split_data')
create_dir(split_dir)

pp_dir = os.path.join(data_dir, 'preprocessing_data')
create_dir(pp_dir)

atom_cutoff = 60
atom_length = 127
bond_length = 12

result_dir = os.path.join(par_dir, 'result')
create_dir(result_dir)

log_dir = os.path.join(par_dir, 'logs')
create_dir(log_dir)

dense_size = 256
task_dense_size = 128
dropout = 0.3
learning_rate = 1e-3

epochs = 1000 
batch_size = 256
patience = 200
