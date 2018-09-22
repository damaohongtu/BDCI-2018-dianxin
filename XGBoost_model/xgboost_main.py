#!/usr/bin/env python
#coding=utf-8
import sys
sys.path.append("/home/tm/workplace/BDCI")
import xgboost as xgb
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import time
import threading

train_data=pd.read_csv('../data/preprocessed_train.csv')
label2current_service = dict(zip(range(0,len(set(train_data['current_service']))),sorted(list(set(train_data['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train_data['current_service']))),range(0,len(set(train_data['current_service'])))))
train_data=pd.read_csv('../data/preprocessed_train.csv')
train_data['current_service']=train_data['current_service'].map(current_service2label)

Y=train_data['current_service']
X=train_data.drop(['current_service','user_id'],axis=1)
X,Y=X.values,Y.values

params={
    'max_depth':12,
    'learning_rate':0.05,
    'n_estimators':752,
    'silent':True,
    'objective':"multi:softmax",
    'nthread':4,
    'gamma':0,
    'max_delta_step':0,
    'subsample':1,
    'colsample_bytree':0.9,
    'colsample_bylevel':0.9,
    'reg_alpha':1,
    'reg_lambda':1,
    'scale_pos_weight':1,
    'base_score':0.5,
    'seed':2018,
    'missing':None,
    'num_class':15
}

plst = list(params.items())
num_rounds = 5000 # 迭代次数

start_time=time.time()
test_data=pd.read_csv('../data/preprocessed_test.csv')
user_id=test_data.pop('user_id')
test_data=test_data.values
xgb_test=xgb.DMatrix(test_data)

from sklearn.model_selection import StratifiedKFold
n_splits = 5
seed = 42
skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)

def run(index,train_index,test_index):
    print(index)
    X_train, X_valid, y_train, y_valid = X[train_index],X[test_index],Y[train_index],Y[test_index]
    xgb_train=xgb.DMatrix(X_train,label=y_train)
    xgb_val=xgb.DMatrix(X_valid,y_valid)
    watchlist = [(xgb_val, 'val')]
    model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=50)
    model.save_model('./xgb%s.model'%str(index))

def main():
    threads=[]
    for index,(train_index,test_index) in enumerate(skf.split(X,Y)):
        t=threading.Thread(target=run,args=(index,train_index,test_index,))
        print("start thread:%d"%index)
        t.start()
        threads.append(t)
    for k in threads:
        k.join()

if __name__ == '__main__':
    start_time=time.time()
    main()
    cv_pred=[]
    for index in range(0,5):
        bst=xgb.Booster({'nthread':4})
        bst.load_model('./xgb%s.model'%str(index))
        y_test = np.asarray(bst.predict(xgb_test),dtype=np.int)
        if index == 0:
            cv_pred = np.array(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))
    # 投票,将数据分成五份，用五份数据训练五个模型，每个模型对test进行预测，预测结果
    submit = []
    print(cv_pred[0:5])
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))
    result=pd.DataFrame(submit,columns=['predict'])
    result['predict']=result['predict'].map(label2current_service)
    xgb_submission=pd.concat([user_id,result],axis=1)
    xgb_submission.to_csv('./xgb_submission.csv',index=False)

    print("over! time cost:%s"%str(time.time()-start_time))