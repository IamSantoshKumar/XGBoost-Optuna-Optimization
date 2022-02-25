import os
import argparse
import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
import xgboost as xgb
import enum
import math
import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def optimize(trial,x,y):
    n_estimators=trial.suggest_int('n_estimators',100,20000)
    max_depth=trial.suggest_int('max_depth',2,15)
    min_child_weight=trial.suggest_int('min_child_weight',1,500)
    learning_rate=trial.suggest_loguniform('learning_rate',0.01, 1.0)
    reg_gamma=trial.suggest_float('reg_gamma',0.1, 30.0)
    reg_alpha=trial.suggest_float('reg_alpha',0.1, 30.0)
    reg_lambda=trial.suggest_float('reg_lambda',0.1, 30.0)
    subsample=trial.suggest_float('subsample',0.01,1.0,log=True)
    colsample_bytree=trial.suggest_float('colsample_bytree',0.01,1.0,log=True)
    
    xgb_params= {
        "objective": "binary:logistic",
        'eval_metric': 'auc',
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        "reg_alpha" : reg_alpha,
        "reg_lambda": reg_lambda,
        "gamma" :reg_gamma,
        "min_child_weight": min_child_weight,
        "n_jobs": 4,
        "seed": args.seed,
        'tree_method': "gpu_hist",
        "gpu_id": 0,
    }
    
    num_rounds=args.n_iters
	
    auc_score = []
        
    kf = model_selection.StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    for f, (train_idx, val_idx) in tqdm(enumerate(kf.split(x, y))):
        df_train, df_val = x.iloc[train_idx], x.iloc[val_idx]
        train_target, val_target = y[train_idx], y[val_idx]
        train = xgb.DMatrix(df_train, label=train_target)
        valid = xgb.DMatrix(df_val, label=val_target)
        watchlist  = [ (train,'train'),(valid,'eval')]
		
        model = xgb.train(
		    xgb_params, 
		    train, 
		    num_rounds,
		    watchlist,
		    early_stopping_rounds=args.es_stop, 
	)
        predicted = model.predict(valid)
        auc  = roc_auc_score(val_target, predicted)
        auc_score.append(auc)
      
    return np.mean(auc_score)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--path", type=str, default="../")
    parser.add_argument("--filename", type=str, default="train.csv")
    parser.add_argument("--n_trials", type=float, default=100, required=False)
    parser.add_argument("--es_stop", type=int, default=100, required=False)
    parser.add_argument("--n_iters", type=int, default=200, required=False)
    
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    df = pd.read_csv(os.path.join(args.path, args.filename))
    optimize_func=partial(optimize,x=df, y=df['target'].values)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_func, n_trials=args.n_trials)
    trial = study.best_trial
    print('Score: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
