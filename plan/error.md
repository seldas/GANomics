>>> [cpu] Starting: NB_Ablation_Size_50_Run_0
Using device: cpu
Traceback (most recent call last):
  File "/compute001/lwu/projects/GANomics/scripts/train.py", line 186, in <module>
    train()
    ~~~~~^^
  File "/compute001/lwu/projects/GANomics/scripts/train.py", line 69, in train
    dataset = GenomicsDataset(
        config['dataset']['path_A'],
    ...<4 lines>...
        force_index_mapping=config['dataset'].get('force_index_mapping', True)
    )
  File "/compute001/lwu/projects/GANomics/src/datasets/genomics_dataset.py", line 46, in __init__
    self.df_B = self.df_B.loc[all_samples]
                ~~~~~~~~~~~~~^^^^^^^^^^^^^
  File "/compute001/lwu/projects/GANomics/venv/lib/python3.13/site-packages/pandas/core/indexing.py", line 1207, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)
           ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/compute001/lwu/projects/GANomics/venv/lib/python3.13/site-packages/pandas/core/indexing.py", line 1438, in _getitem_axis
    return self._getitem_iterable(key, axis=axis)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/compute001/lwu/projects/GANomics/venv/lib/python3.13/site-packages/pandas/core/indexing.py", line 1378, in _getitem_iterable
    keyarr, indexer = self._get_listlike_indexer(key, axis)
                      ~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/compute001/lwu/projects/GANomics/venv/lib/python3.13/site-packages/pandas/core/indexing.py", line 1576, in _get_listlike_indexer
    keyarr, indexer = ax._get_indexer_strict(key, axis_name)
                      ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/compute001/lwu/projects/GANomics/venv/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 6302, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
    ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/compute001/lwu/projects/GANomics/venv/lib/python3.13/site-packages/pandas/core/indexes/base.py", line 6352, in _raise_if_missing
    raise KeyError(f"None of [{key}] are in the [{axis_name}]")
KeyError: "None of [Index(['UKv4_A_23_P3527', 'UKv4_A_24_P82749', 'UKv4_A_23_P106103',\n       'UKv4_A_23_P214950', 'UKv4_A_23_P324989', 'UKv4_A_24_P208704',\n       'UKv4_A_24_P270496', 'UKv4_A_23_P418597', 'UKv4_A_23_P96812',\n       'UKv4_A_23_P24269',\n       ...\n       'UKv4_A_32_P187176', 'UKv4_A_23_P101759', 'UKv4_A_23_P207493',\n       'UKv4_A_23_P8142', 'UKv4_A_23_P17330', 'UKv4_A_23_P11032',\n       'UKv4_A_23_P106362', 'UKv4_A_23_P81131', 'UKv4_A_23_P120566',\n       'UKv4_A_23_P162378'],\n      dtype='str', name='samples_name', length=10042)] are in the [index]"