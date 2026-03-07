import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random

class GenomicsDataset(Dataset):
    """
    Dataset for genomic expression data (Microarray and RNA-seq).
    Supports loading from CSV/TSV files and provides paired/unpaired sampling.
    """
    def __init__(self, path_A, path_B, is_train=True, max_samples=None, random_seed=None):
        """
        path_A: Path to domain A (e.g. Microarray) CSV/TSV
        path_B: Path to domain B (e.g. RNA-seq) CSV/TSV
        is_train: Whether in training mode (affects sampling)
        random_seed: Optional seed for reproducible shuffling of samples
        """
        self.is_train = is_train
        
        # Load data (assuming CSV with index as sample IDs and columns as genes)
        self.df_A = self._load_df(path_A)
        self.df_B = self._load_df(path_B)
        
        # Ensure genes are aligned
        common_genes = self.df_A.columns.intersection(self.df_B.columns)
        self.df_A = self.df_A[common_genes]
        self.df_B = self.df_B[common_genes]
        
        # Randomize samples if seed provided
        all_samples = self.df_A.index.tolist()
        if random_seed is not None:
            random.Random(random_seed).shuffle(all_samples)
            self.df_A = self.df_A.loc[all_samples]
            self.df_B = self.df_B.loc[all_samples]

        if max_samples:
            self.df_A = self.df_A.head(max_samples)
            self.df_B = self.df_B.head(max_samples)
            
        self.samples_A = self.df_A.index.tolist()
        self.samples_B = self.df_B.index.tolist()
        
        # For paired data, we assume the index (sample ID) matches
        self.common_samples = sorted(list(set(self.samples_A) & set(self.samples_B)))
        
        self.A_size = len(self.samples_A)
        self.B_size = len(self.samples_B)

    def _load_df(self, path):
        if path.endswith('.csv'):
            return pd.read_csv(path, index_col=0)
        elif path.endswith('.tsv') or path.endswith('.txt'):
            return pd.read_csv(path, sep='\t', index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {path}")

    def __len__(self):
        return max(self.A_size, self.B_size)

    def __getitem__(self, index):
        # A: Sequential sample from A
        sample_A_id = self.samples_A[index % self.A_size]
        val_A = torch.tensor(self.df_A.loc[sample_A_id].values, dtype=torch.float32)
        
        # B: Random or sequential sample from B (for unpaired/adversarial loss)
        if self.is_train:
            index_B = random.randint(0, self.B_size - 1)
        else:
            index_B = index % self.B_size
        sample_B_id = self.samples_B[index_B]
        val_B = torch.tensor(self.df_B.loc[sample_B_id].values, dtype=torch.float32)
        
        # Paired samples for Feedback Loss
        # paired_B is the counterpart of sample_A_id in domain B
        if sample_A_id in self.df_B.index:
            paired_B = torch.tensor(self.df_B.loc[sample_A_id].values, dtype=torch.float32)
        else:
            # Fallback if not paired (should not happen in GANomics training set)
            paired_B = val_B 
            
        # paired_A is the counterpart of sample_B_id in domain A
        if sample_B_id in self.df_A.index:
            paired_A = torch.tensor(self.df_A.loc[sample_B_id].values, dtype=torch.float32)
        else:
            paired_A = val_A

        return {
            'A': val_A, 
            'B': val_B, 
            'medianA': paired_A, 
            'medianB': paired_B,
            'A_path': sample_A_id,
            'B_path': sample_B_id
        }
