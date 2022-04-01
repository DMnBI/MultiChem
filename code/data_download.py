import os
import deepchem as dc

import data_config as cf 

if __name__ == '__main__':
    moleculeNet_tox21_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
    moleculeNet_toxcast_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"
    moleculeNet_sider_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"
    moleculeNet_hiv_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"

    dc.utils.data_utils.download_url(url=moleculeNet_tox21_url, dest_dir=cf.origin_dir)
    dc.utils.data_utils.download_url(url=moleculeNet_toxcast_url, dest_dir=cf.origin_dir)
    dc.utils.data_utils.download_url(url=moleculeNet_sider_url, dest_dir=cf.origin_dir)
    dc.utils.data_utils.download_url(url=moleculeNet_hiv_url, dest_dir=cf.origin_dir)
