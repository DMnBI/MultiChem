import os
import sys
import pandas as pd
import numpy as np

import data_config as cf 
from rdkit import Chem

def get_mol_from_smile(smile):
    if smile == 'FAIL':
        return Chem.Mol()
    else:
        mol = Chem.MolFromSmiles(smile, sanitize=False)
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol,\
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|\
                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|\
                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|\
                Chem.SanitizeFlags.SANITIZE_SYMMRINGS,\
                catchErrors=True)
        return mol

def one_hot_encoding(x, check_list): 
    return list(map(lambda s: float(x == s), check_list))

def atom_feature(atom):
    return np.array(
            one_hot_encoding(int(atom.GetAtomicNum()), list(range(1,101))) +
            one_hot_encoding(int(atom.GetTotalDegree()), list(range(0,6))) +
            one_hot_encoding(int(atom.GetFormalCharge()), list(range(-2,3))) +
            one_hot_encoding(str(atom.GetChiralTag()), ['CHI_UNSPECIFIED','CHI_TETRAHEDRAL_CW','CHI_TETRAHEDRAL_CCW','CHI_OTHER']) +
            one_hot_encoding(int(atom.GetTotalNumHs()), list(range(0,5))) +
            one_hot_encoding(str(atom.GetHybridization()), ['SP','SP2','SP3','SP3D','SP3D2']) +
            [float(atom.GetIsAromatic())] +
            [float(atom.GetMass())*0.01]
            )

def bond_feature(bond):
    return np.array(
            one_hot_encoding(str(bond.GetBondType()), ['SINGLE','DOUBLE','TRIPLE','AROMATIC']) +
            [float(bond.GetIsConjugated())] +
            [float(bond.IsInRing())] +
            one_hot_encoding(str(bond.GetStereo()), ['STEREONONE','STEREOANY','STEREOE','STEREOZ','STEREOCIS','STEREOTRANS'])
            )

def bond_index(bond):
    return [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]

def atom_feature_list(mol):
    atoms = mol.GetAtoms() 
    features = np.zeros((cf.atom_cutoff, cf.atom_length))
    for i, f in enumerate(list(map(atom_feature, atoms))):
        if i >= cf.atom_cutoff:
            break
        features[i] = f
    return features

def bond_feature_list(mol):
    bonds = mol.GetBonds() 
    count = np.zeros(cf.atom_cutoff)
    features = np.zeros((cf.atom_cutoff, cf.atom_cutoff, 2*cf.bond_length))
    for f, (i,j) in zip(list(map(bond_feature, bonds)), list(map(bond_index, bonds))):
        if i >= cf.atom_cutoff or j >= cf.atom_cutoff:
            continue
        else:
            if count[i] < cf.bond_cutoff:
                features[i][j] = np.pad(f, (0,cf.bond_length), 'constant', constant_values=0)
                count[i] += 1
            if count[j] < cf.bond_cutoff:
                features[j][i] = np.pad(f, (cf.bond_length,0), 'constant', constant_values=0)
                count[j] += 1
    return features

def graph_feature_list(mol):
    bonds = mol.GetBonds() 
    count = np.zeros(cf.atom_cutoff)
    graph = np.zeros((cf.atom_cutoff, cf.atom_cutoff))
    for (i,j) in list(map(bond_index, bonds)):
        if i >= cf.atom_cutoff or j >= cf.atom_cutoff:
            continue
        else:
            if count[i] < cf.bond_cutoff:
                graph[i][j] = 1
                count[i] += 1
            if count[j] < cf.bond_cutoff:
                graph[j][i] = 1
                count[j] += 1
    return graph

def get_clear_labels(labels):
    for task in labels.columns:
        labels[task] = labels[task].map({'0':0, '1':1, 'x':-1, 0:0, 1:1})
    labels = labels.fillna(-1).to_numpy(dtype=np.float32)
    return labels

def preprocessing(name, mode, tasks=None):
    data = pd.read_csv(os.path.join(cf.split_dir, name+'_'+mode+'.csv'))
    mols = list(map(get_mol_from_smile, data['SMILES']))

    features_a = np.array(list(map(atom_feature_list, mols)))
    features_b = np.array(list(map(bond_feature_list, mols)))
    features_g = np.array(list(map(graph_feature_list, mols)))

    if tasks:
        labels = get_clear_labels(data.loc[:, tasks].copy())
    else:
        labels = get_clear_labels(data.loc[:, ~data.columns.isin(['ID', 'SMILES'])].copy())

    np.savez_compressed(os.path.join(cf.pp_dir, name+'_'+mode),
            atom = features_a, bond = features_b, graph = features_g, label = labels)

def preprocessing_data(name, tasks=None):
    preprocessing(name, 'train', tasks)
    preprocessing(name, 'valid', tasks)
    preprocessing(name, 'test', tasks)

if __name__ == '__main__':
    preprocessing_data('tox21', ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD','NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])
    #preprocessing_data('toxcast')
    #preprocessing_data('sider')
    #preprocessing_data('hiv')
    #preprocessing_data('bbbp')
    #preprocessing_data('clintox')
