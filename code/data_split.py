import os
import sys
import pandas as pd
import numpy as np

import data_config as cf 

def random_split(dataset):
    return np.split(dataset.sample(frac=1), [int(0.8*len(dataset)), int(0.9*len(dataset))])

def split_and_save(dataset, name, mode='random'):
    if mode == 'random':
        train, valid, test = random_split(dataset) 
        train.to_csv(os.path.join(cf.split_dir, name+'_train.csv'), index=False)
        valid.to_csv(os.path.join(cf.split_dir, name+'_valid.csv'), index=False)
        test.to_csv(os.path.join(cf.split_dir, name+'_test.csv'), index=False)
    else:
        print('split mode error')
        exit()

if __name__ == '__main__':
    dataset = pd.read_csv(os.path.join(cf.origin_dir, 'tox21.csv.gz'), compression='gzip')
    dataset.rename(columns={'mol_id':'ID', 'smiles':'SMILES'}, inplace=True)
    split_and_save(dataset, 'tox21')

    dataset = pd.read_csv(os.path.join(cf.origin_dir, 'toxcast_data.csv.gz'), compression='gzip')
    dataset.rename(columns={'smiles':'SMILES'}, inplace=True)
    dataset['ID'] = np.arange(1, 1+len(dataset)).astype('str')
    split_and_save(dataset, 'toxcast')

    dataset = pd.read_csv(os.path.join(cf.origin_dir, 'sider.csv.gz'), compression='gzip')
    dataset.rename(columns={'smiles':'SMILES'}, inplace=True)
    dataset['ID'] = np.arange(1, 1+len(dataset)).astype('str')
    split_and_save(dataset, 'sider')

    dataset = pd.read_csv(os.path.join(cf.origin_dir, 'HIV.csv'))
    dataset.rename(columns={'smiles':'SMILES'}, inplace=True)
    dataset['ID'] = np.arange(1, 1+len(dataset)).astype('str')
    dataset.drop('activity', axis=1, inplace=True)
    split_and_save(dataset, 'hiv')
