if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument('--train_file', default='/home/database')
	parser.add_argument('--val_file', default='/home/database')
	parser.add_argument('--test_file', default='/home/database')
	parser.add_argument('--batch_size', default=256, type=int)
	parser.add_argument('--num_workers', default=4, type=int)

	parser.add_argument('--node_size', default=127, type=int)
	parser.add_argument('--layer_size', default=128, type=int)
	parser.add_argument('--layer_depth', default=3, type=int)
	parser.add_argument('--label_size', default=12, type=int)
	parser.add_argument('--dropout', default=0.5, type=float)
	parser.add_argument('--edge_size', default=12, type=int)
	parser.add_argument('--heads', default=4, type=int)

	parser.add_argument('--learning_rate', default=1e-3, type=float)
	parser.add_argument('--decay', default=0.0, type=float)

	parser.add_argument('--log_dir', default='/home/MultiChem/log')
	parser.add_argument('--patience', default=50, type=int)
	parser.add_argument('--epoch', default=5000, type=int)
	parser.add_argument('--gpus', default=[3], nargs='+', type=int)

	parser.add_argument('--learning', action='store_true')
	parser.add_argument('--predict', action='store_true')

	args = parser.parse_args()

	import os

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(i) for i in args.gpus)

	args_dict = vars(args)

	from multi_chem.learning.learner import learn_MultiChem

	learner = learn_MultiChem(**args_dict)
	learner.run()
