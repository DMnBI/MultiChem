<!-- #region -->
# MultiChem: predicting chemical properties using multi-view graph attention network
-- -- --

## Environment
- - -
- Conda 24.7.1 
- CUDA 11.3

## Requirements
- - -
```bash
conda env create --file requirements.yaml 
```

## Example
- - -
**Create 30 seeds**
```python
python ./example_data/thirty_data/make_dataset.py tox21
```

**Training**
```python
python run.py --train_file ./example_data/thirty_data/tox21/fold_0/train.csv --val_file ./example_data/thirty_data/tox21/fold_0/valid.csv --test_file ./example_data/thirty_data/tox21/fold_0/test.csv --log_dir ./Log_thirty/tox21/fold_0 --batch_size 256 --label_size 12 --gpus 0 --learning --task_type 0
```

**Prediction**
```python
python run.py --train_file ./example_data/thirty_data/tox21/fold_0/train.csv --val_file ./example_data/thirty_data/tox21/fold_0/valid.csv --test_file ./example_data/thirty_data/tox21/fold_0/test.csv --log_dir ./Log_thirty/tox21/fold_0 --batch_size 256 --label_size 12 --gpus 0 --predict --task_type 0
```

<!-- #endregion -->
