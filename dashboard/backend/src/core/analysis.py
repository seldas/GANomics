import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, precision_score, f1_score

def run_deg_analysis(df, labels, group1=1, group2=0):
    """
    Perform vectorized Welch's t-test for DEGs.
    """
    # Ensure labels are compared correctly (handles both numeric and string representations)
    g1_data = df.loc[labels.astype(float) == float(group1)]
    g2_data = df.loc[labels.astype(float) == float(group2)]

    n1 = len(g1_data)
    n2 = len(g2_data)
    
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Insufficient samples for DEG analysis: group1={n1}, group2={n2}")

    # Vectorized Means and Variances
    m1 = g1_data.mean(axis=0)
    m2 = g2_data.mean(axis=0)
    v1 = g1_data.var(axis=0, ddof=1)
    v2 = g2_data.var(axis=0, ddof=1)

    # Welch's T-test formula components
    se = np.sqrt(v1/n1 + v2/n2)
    # Handle zero SE to avoid division by zero
    se = np.where(se == 0, np.nan, se)
    
    t_stats = (m1 - m2) / se
    
    # Degrees of Freedom (Satterthwaite)
    df_num = (v1/n1 + v2/n2)**2
    df_den = (v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)
    dof = df_num / df_den
    
    # P-values
    p_values = 2 * stats.t.sf(np.abs(t_stats), dof)

    # Cohen's d (Pooled SD)
    s_pooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    s_pooled = np.where(s_pooled == 0, np.nan, s_pooled)
    cohen_ds = (m1 - m2) / s_pooled

    results = pd.DataFrame({
        'gene': df.columns,
        't_stat': t_stats,
        'p_value': p_values,
        'cohen_d': cohen_ds
    })

    # Benjamini-Hochberg correction
    results = results.sort_values('p_value')
    results['fdr'] = results['p_value'] * len(results) / np.arange(1, len(results) + 1)
    results['fdr'] = np.minimum.accumulate(results['fdr'][::-1])[::-1]
    results['fdr'] = np.minimum(results['fdr'], 1.0)
    
    results.set_index('gene', inplace=True)
    
    return results

def train_eval_rf(train_x, train_y, test_x, test_y):
    """
    Train and evaluate a Random Forest classifier.
    """
    # Ensure labels are strings for consistent evaluation
    train_y = train_y.astype(str)
    test_y = test_y.astype(str)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_x, train_y)
    
    preds = rf.predict(test_x)
    
    # Find the '0' class (Unfavorable) or use the first class if missing
    classes = rf.classes_
    pos_label = '0' if '0' in classes else classes[0]
    
    return {
        'MCC': float(matthews_corrcoef(test_y, preds)),
        'Accuracy': float(accuracy_score(test_y, preds)),
        'Recall': float(recall_score(test_y, preds, pos_label=pos_label, zero_division=0)),
        'Precision': float(precision_score(test_y, preds, pos_label=pos_label, zero_division=0)),
        'F1': float(f1_score(test_y, preds, pos_label=pos_label, zero_division=0))
    }
