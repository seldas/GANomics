import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score

def run_deg_analysis(df, labels, group1='Favorable', group2='Unfavorable'):
    """
    Perform per-gene Welch's t-test for DEGs.
    """
    g1_data = df.loc[labels == group1]
    g2_data = df.loc[labels == group2]
    
    p_values = []
    t_stats = []
    
    for gene in df.columns:
        t, p = stats.ttest_ind(g1_data[gene], g2_data[gene], equal_var=False)
        p_values.append(p)
        t_stats.append(t)
        
    results = pd.DataFrame({
        'gene': df.columns,
        't_stat': t_stats,
        'p_value': p_values
    })
    
    # Benjamini-Hochberg correction
    results = results.sort_values('p_value')
    results['fdr'] = results['p_value'] * len(results) / np.arange(1, len(results) + 1)
    results['fdr'] = np.minimum.accumulate(results['fdr'][::-1])[::-1]
    results['fdr'] = np.minimum(results['fdr'], 1.0)
    
    return results

def train_eval_rf(train_x, train_y, test_x, test_y):
    """
    Train and evaluate a Random Forest classifier.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y)
    
    preds = rf.predict(test_x)
    
    return {
        'MCC': matthews_corrcoef(test_y, preds),
        'Accuracy': accuracy_score(test_y, preds),
        'Recall': recall_score(test_y, preds, pos_label='Unfavorable'),
        'Precision': precision_score(test_y, preds, pos_label='Unfavorable'),
        'F1': f1_score(test_y, preds, pos_label='Unfavorable')
    }

def cross_platform_evaluation(real_train_x, real_train_y, 
                             syn_test_x, real_test_y):
    """
    Scenario Q1: Train on Real -> Predict on Synthetic
    """
    return train_eval_rf(real_train_x, real_train_y, syn_test_x, real_test_y)
