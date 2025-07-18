#EDA
统计摘要：df.describe(), df.info() 用于数据概览（数据类型、缺失、分布）；你已有这步。

分布可视化：通过 df.hist()、sns.boxplot() 快速可视化数值列分布与异常值。

类别变量分析：用 sns.countplot(x='cat_col', data=df) 展示类别分布。

相关性分析：使用 df.corr() 和 sns.heatmap() 看各变量间的相关程度。
模块化
.
├── data_processing.py       # 特征工程、清洗函数
├── model_training.py        # 模型训练 + early stopping + save model
├── model_evaluation.py      # AUC、KS等评估函数
├── model_predict.py         # 批量打分
├── api_service.py           # FastAPI 或 Flask 接口
├── config.py                # 模型超参、路径等
├── main.py                  # 脚本入口：组合 ETL、训练、评估、部署等流程
├── models/
│   └── model_v1.pkl         # 模型 artifact 存放
└── tests/                   # （可选）测试样例或 mock 数据

#data_proc.py
import pandas as pd
import numpy as np
import json

def readindata( datapath):
    df =   spark.read(datapath)
    print(df.info())
    print(df.decribe())  
    return df
def preprocess(df: pd.DataFrame, onehot_col: str,  drop_cols: list, fe_config_path: str = "fe_config.json"):
    numcol = df.select_dtypes(include=['float','int']).columns.tolist()
    catcol = df.select_dtype(include=['object']).columns.tolist()
    median = df[numcol].median()
    mode = df[catcol].mode()
    df[numcol]=df[numcol].fillna(median)
    df[catcol]=df[catcol].fillna(mode)
    #onehot
    df  = pd.get_dummies(df,drop_cols,prefix = 'xxx')
    ohe_cols = [c for c in df.columns if c.startswith('xxx')]
    df = df.drop[drop_cols]
    df['new'] = df.old1/df.old2
    fe_config = {
        "numeric_cols": numcol,
        "categorical_cols": catcol,
        "median": median,
        "mode": mode,
        "ohe_columns": ohe_cols,
        "custom_features": ['new'],
        "drop_columns": drop_cols,
        "onehot_col": onehot_col
    }
    with open(fe_config_path,'w') as f:
        json.dump(fe_config,f,indent=2)

    return df  
#高级imputer

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[num_cols] = imputer.fit_transform(df[num_cols])

#same process when inference: fe means saved model using joblit, the feature engineer part of the work also saved into fe

def preprocess_new(df, fe_config):
    if 'old' in df.columns and 'old2' in df.columns and 'new' in fe_config.get('custom_features', []):
        df['new'] = df['old'] / df['old2']

    df[fe_config['numeric_cols']] = df[fe_config['numeric_cols']].fillna(fe_config['median'])
    df = pd.get_dummies(df, columns=fe_config['categorical_cols'])

    for col in fe_config['ohe_columns']:
        if col not in df.columns:
            df[col] = 0
    df = df[fe_config['ohe_columns']]

    return df
#训练集 / 验证集划分
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, label_col='target'):
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def find_best_hyperparams(X_train, y_train  ):
    base_model = XGBClassifier(use_label_encoder=False, eval_metric='auc')
    
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 300],
        'subsample': [0.8],
        'colsample_bytree': [0.8]
    }

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    
    print("Best AUC:", grid.best_score_)
    print("Best Params:", grid.best_params_)
    
    return grid.best_params_
from xgboost import XGBClassifier

def train_model(X_train, y_train, X_val, y_val,best_params):
    model = XGBClassifier(
        # n_estimators=500,
        # learning_rate=0.05,
        # max_depth=4,
        # subsample=0.8,
        # colsample_bytree=0.8,
        **best_params,
        use_label_encoder=False,
        eval_metric='auc'
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10,
        verbose=False
    )
    featureimp = model.feature_importance_
    return model,featureimp

import matplotlib.pyplot as plt
plt.figure()
plt.barh(X_train.columns, model.feature_importances_)
plt.title('Feature Importances')
plt.show()


#模型评估（AUC + KS）
from sklearn.metrics import roc_auc_score

def compute_ks(y_true, y_score, bins=100):
    df = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    df = df.sort_values(by='y_score', ascending=False)
    df['bucket'] = pd.qcut(df['y_score'], q=bins, duplicates='drop')
    grouped = df.groupby('bucket')['y_true'].agg(['count', 'sum'])
    grouped = grouped.rename(columns={'count': 'total', 'sum': 'positive'})
    grouped['negative'] = grouped['total'] - grouped['positive']
    grouped['cum_pos'] = grouped['positive'].cumsum() / grouped['positive'].sum()
    grouped['cum_neg'] = grouped['negative'].cumsum() / grouped['negative'].sum()
    return (grouped['cum_pos'] - grouped['cum_neg']).abs().max()

def eval bbbbbbccxxfate_model(model, X_val, y_val):
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    ks = compute_ks(y_val, y_pred_proba)
    print(f"AUC: {auc:.4f}, KS: {ks:.4f}")
    return auc, ks
#IF IT'S REGRESSION
from sklearn.metrics import mean_squared_error, r2_score
print("RMSE:", mean_squared_error(y_true, y_pred, squared=False))
print("R2:", r2_score(y_true, y_pred))

def predict_and_score(model, X_new):
    proba = model.predict_proba(X_new)[:, 1]
    return proba

#存取超参

import json
import os

def load_best_params(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None
def save_best_params(params, config_path):
    with open(config_path, 'w') as f:
        json.dump(params, f)

#存feature importance在model metadaata folder里
def save_feature_importance(modelpath,featurenames,featureimp):
    data = [{'feature':f, 'importance':float(i)}for f, i in zip(featurenames, featureimp)]
    with open(modelpath,'w') as f:
        json.dump(data,f,indent=2)
import json
def main():
    # Step 1: 读取数据
    df = pd.read_csv('loan_data.csv')  # 必须包含 'target' 列

    # Step 2: 特征处理
    df_clean, fe_config = preprocess_data(df)

    # Step 3: 划分训练集
    X_train, X_val, y_train, y_val = split_data(df_clean, label_col='target')

    # Step 4: 模型训练

     #查看超参是否已经存咋
      
    best_params = load_best_params
    if best_params is None: 
        best_params = find_best_hyperparams(X_train, y_train, X_val, y_val)
        save_best_params(best_params, 'config/best_params.json') 
  
    model = train_model(X_train, y_train, X_val, y_val,**best_params)

    # Step 5: 模型评估
    evaluate_model(model, X_val, y_val)

    # Step 6: 模型评分新数据（例子）
    new_data = df_clean.sample(5).drop(columns='target')
    scores = predict_and_score(model, new_data)
    print("Example scores:", scores)
import joblib

# 保存模型和特征工程步骤（如果有）
joblib.dump({
    'model': model,
    'fe_config': fe_config
}, 'model_bundle.pkl')


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Step 1: 加载模型
model_bundle = joblib.load('model_bundle.pkl')
model = model_bundle['model']
columns = model_bundle['columns']

# Step 2: 定义请求格式
class LoanRequest(BaseModel):
    annual_income: float
    loan_amount: float
    credit_score: float
    loan_purpose: str  # car/home/education etc.

# Step 3: 初始化 app
app = FastAPI()

# Step 4: 实时评分端点
@app.post("/score")#trigger, that request data coming in  and API pass to model and return score
def score_loan(data: LoanRequest):
    # 构建 DataFrame
    df = pd.DataFrame([{
        'annual_income': data.annual_income,
        'loan_amount': data.loan_amount,
        'credit_score': data.credit_score,
        'loan_purpose': data.loan_purpose
    }])
    
    # 特征工程：用fe config
    for col in fe_config['numeric_cols']:
        if df.col.isnull().any():
            df[col] = df.fillna(fe_config["median"][col])
    df = df[fe_config['numeric_cols']_fe_config['']]
    ...
    for col in fe_config['ohe_colunm']:
        if col not in df.columns:
     
 
    
    # 预测概率
    score = model.predict_proba(df)[:, 1][0]
    return {"score": float(score)}


import joblib
# 假设模型训练完后存在变量 model 中
joblib.dump({
    'model': model,
    'fe_config': fe_config
}, 'models/model_bundle.pkl')

#shadow打分不返回
@app.post("/score")
def score(data: LoanRequest):
    features = preprocess(data) #错的要用fe_config里面存的median mode onehotvalues to process data

    score_v1 = model_v1.predict_proba(features)[:, 1]
    score_v2 = model_v2.predict_proba(features)[:, 1]  # shadow

    log_to_s3({
        "user": data.user_id,
        "score_v1": score_v1,
        "score_v2": score_v2
    })

    return {"score": float(score_v1)}  # 返回旧模型分数
#10%上新模型
import random

@app.post("/score")
def score(data: LoanRequest):
    p = random.random()
    if p < 0.1:
        model = model_v2  # canary
    else:
        model = model_v1
    ...


