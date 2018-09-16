#!/usr/bin/env python
#coding=utf-8
import pandas as pd
from catboost import CatBoostClassifier, Pool, cv
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

train_data=pd.read_csv('../data/preprocessed_train.csv')

names=list((train_data.head(0)))
cat_names=['service_type', 'is_mix_service', 'many_over_bill', 'contract_type','is_promise_low_consume','net_service', 'gender', 'complaint_level']
categorical_features_indices=[names.index(a) for a in cat_names]

label2current_service = dict(zip(range(0,len(set(train_data['current_service']))),sorted(list(set(train_data['current_service'])))))
current_service2label = dict(zip(sorted(list(set(train_data['current_service']))),range(0,len(set(train_data['current_service'])))))
train_data=pd.read_csv('../data/preprocessed_train.csv')
train_data['current_service']=train_data['current_service'].map(current_service2label)

Y=train_data['current_service']
X=train_data.drop(['current_service','user_id'],axis=1)

# seed = 7
# test_size = 0.33
# X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=test_size, random_state=seed)

params = {
    'loss_function': 'MultiClass',
    'iterations': 5000,
    'l2_leaf_reg':15,
    'depth':5,
    'learning_rate': 0.05,
    'eval_metric': 'TotalF1',
    'random_seed': 42,
    'classes_count':15,
}


print("start catboost...")

from sklearn.model_selection import StratifiedKFold
n_splits = 5
seed = 42
test_data=pd.read_csv('../data/preprocessed_test.csv')
user_id=test_data.pop('user_id')
cb_test=Pool(test_data)

skf = StratifiedKFold(n_splits=n_splits,random_state=seed,shuffle=True)
X,Y=X.values,Y.values
cv_pred = []
for index,(train_index,test_index) in enumerate(skf.split(X,Y)):
    print(index)
    X_train, X_valid, y_train, y_valid = X[train_index],X[test_index],Y[train_index],Y[test_index]
    train_pool = Pool(X_train, y_train)
    validate_pool = Pool(X_valid, y_valid)
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=validate_pool,early_stopping_rounds=50)

    importances = pd.DataFrame({'Feature':model.feature_importances_, 'Importance': model.feature_importances_})
    top_importances = importances.sort_values(by='Importance', ascending=False)[:20]
    fig, ax = plt.subplots(1, 1, figsize=[10, 7])
    sns.barplot(x='Importance', y='Feature', data=top_importances, ax=ax,orient='h')
    plt.tight_layout()

    plt.savefig('feature_importance%s.png'%str(index))
    y_test = np.asarray(model.predict(cb_test),dtype=np.int)
    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

# 投票,将数据分成五份，用五份数据训练五个模型，每个模型对test进行预测，预测结果
submit = []
print(cv_pred[0:5])
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))

#model.fit(X=X,y=Y,cat_features=categorical_features_indices)
print("catboost over...")

result=pd.DataFrame(submit,columns=['predict'])
result['predict']=result['predict'].map(label2current_service)
cb_submission=pd.concat([user_id,result],axis=1)
cb_submission.to_csv('./cb_submission.csv',index=False)

