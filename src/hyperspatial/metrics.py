from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree,distance
from sklearn.metrics import average_precision_score
from analysis.hyperspatial_map_2d_dlpfc.dlpfc_common import pearson_columns

def normalized_emd(true,predicted,coordinates,seed):
    true=np.clip(true,0,None);predicted=np.clip(predicted,0,None)
    if true.sum()<=0 or predicted.sum()<=0:return np.nan
    rng=np.random.default_rng(seed);n=min(192,len(coordinates))
    a=rng.choice(len(coordinates),n,replace=True,p=true/true.sum());b=rng.choice(len(coordinates),n,replace=True,p=predicted/predicted.sum())
    costs=distance.cdist(coordinates[a],coordinates[b]);rows,cols=linear_sum_assignment(costs)
    diameter=max(float(np.linalg.norm(coordinates.max(0)-coordinates.min(0))),1e-8)
    return float(costs[rows,cols].mean()/diameter)

def evaluate_matrices(truth,prediction,registered,coordinates,thresholds=None,seed=42):
    """Frozen reconstruction metrics, evaluated only after prediction sealing."""
    p=pearson_columns(truth,prediction)
    residual=pearson_columns(truth-registered,prediction-registered)
    rmse=np.sqrt(np.mean((truth-prediction)**2,axis=0))
    truth_values=np.expm1(truth);predicted_values=np.maximum(np.expm1(prediction),0)
    thresholds=np.asarray(thresholds if thresholds is not None else np.quantile(truth_values,.8,axis=0))
    diameter=max(float(np.linalg.norm(coordinates.max(0)-coordinates.min(0))),1e-8);rows=[]
    for j in range(truth.shape[1]):
        binary=truth_values[:,j]>thresholds[j];prevalence=float(binary.mean());score=predicted_values[:,j]
        auprc=dice=centroid=emd=np.nan
        if binary.sum()>=10 and binary.sum()<len(binary):
            auprc=float(average_precision_score(binary,score))/max(prevalence,1e-8)
            k=int(binary.sum());top=np.zeros(len(binary),bool);top[np.argpartition(score,-k)[-k:]]=True
            dice=float(2*np.sum(top&binary)/max(int(top.sum()+binary.sum()),1))
            tc=np.average(coordinates,axis=0,weights=np.maximum(truth_values[:,j],1e-8));pc=np.average(coordinates,axis=0,weights=np.maximum(score,1e-8))
            centroid=float(np.linalg.norm(tc-pc)/diameter);emd=normalized_emd(truth_values[:,j],score,coordinates,seed+j)
        rows.append({'gene_index':j,'spatial_pearson':p[j],'residual_pearson':residual[j],
            'log1p_rmse':rmse[j],'auprc_enrichment':auprc,'prevalence_matched_dice':dice,
            'centroid_error':centroid,'spatial_emd':emd,'target_prevalence_for_evaluation':prevalence})
    return pd.DataFrame(rows)

def moran(values,coordinates,k=12):
    z=np.asarray(values,float)-np.mean(values);idx=cKDTree(coordinates).query(coordinates,k=min(k+1,len(coordinates)))[1][:,1:]
    den=idx.shape[1]*np.square(z).sum();return np.nan if den<=1e-12 else float((z[:,None]*z[idx]).sum()/den)
