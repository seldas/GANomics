import pandas as pd
import os
import re
import numpy as np
from tqdm import tqdm
import argparse

def process_tcga_dataset(ag_file_1, ag_file_2, rs_file_1, rs_file_2, output_dir):
    # Microarray data
    df_ag1 = pd.read_csv(ag_file_1, sep="\t", comment='#', index_col=0)
    df_ag2 = pd.read_csv(ag_file_2, sep="\t", comment='#', index_col=0)
    df_ag = pd.concat((df_ag1, df_ag2), axis=1)

    # RNA-seq data
    df_rs1 = pd.read_csv(rs_file_1, sep="\t", comment='#', index_col=0)
    df_rs2 = pd.read_csv(rs_file_2, sep="\t", comment='#', index_col=0)
    df_rs = pd.concat((df_rs1, df_rs2), axis=1)

    print(df_ag.shape, df_rs.shape)

    common_id = (set(df_ag.columns) & set(df_rs.columns))
    print(len(common_id))

    df_ag = df_ag.loc[:, list(common_id)]
    df_rs = df_rs.loc[:, list(common_id)]

    gene_ag = [x.upper() for x in df_ag.index]
    df_ag.index = gene_ag

    from collections import Counter
    gene_pool = Counter(gene_ag)

    used_ind_rs, used_ind_ag = [], []
    for ind, d in df_rs.iterrows():
        tmp_gene = re.split(r'\|', ind)[0]
        if gene_pool[tmp_gene.upper()] == 1:
            used_ind_rs.append(ind)
            used_ind_ag.append(tmp_gene.upper())

    df_ag = df_ag.loc[used_ind_ag, :]
    df_rs = df_rs.loc[used_ind_rs, :]

    df_rs = df_rs.transpose()
    df_ag = df_ag.transpose()

    df_rs.columns = df_ag.columns

    print(df_ag.shape, df_rs.shape)

    cols_with_nan_ag = df_ag.columns[df_ag.isnull().any()].tolist()
    cols_with_nan_rs = df_rs.columns[df_rs.isnull().any()].tolist()
    filter_cols = list(set(cols_with_nan_ag + cols_with_nan_rs))

    df_ag_new = df_ag.drop(filter_cols, axis=1)
    df_rs_new = df_rs.drop(filter_cols, axis=1)

    df_ag_new = np.log2(df_ag_new + 1)
    df_rs_new = np.log2(df_rs_new + 1)

    print(df_ag_new.shape, df_rs_new.shape)

    BRCA_samples = [x for x in df_ag_new.index if x in df_ag1.columns]
    with open(f'{output_dir}/BRCA_samples.txt', 'w') as f:
        for x in BRCA_samples:
            f.write(x + '\n')

    df_ag_new.to_csv(f'{output_dir}/COMB_ag.tsv', sep="\t")
    df_rs_new.to_csv(f'{output_dir}/COMB_rs.tsv', sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TCGA dataset")
    parser.add_argument("--ag_file_1", type=str, required=True, help="Path to first Agilent microarray file")
    parser.add_argument("--ag_file_2", type=str, required=True, help="Path to second Agilent microarray file")
    parser.add_argument("--rs_file_1", type=str, required=True, help="Path to first RNA-seq file")
    parser.add_argument("--rs_file_2", type=str, required=True, help="Path to second RNA-seq file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    args = parser.parse_args()

    process_tcga_dataset(args.ag_file_1, args.ag_file_2, args.rs_file_1, args.rs_file_2, args.output_dir)