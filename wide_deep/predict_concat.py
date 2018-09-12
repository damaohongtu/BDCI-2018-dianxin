#!/usr/bin/env python
#coding=utf-8
import pandas as pd
label_vocabulary={0:'99999825', 1:'90063345', 2:'90109916', 3:'89950166', 4:'89950168',
                  5:'99104722', 6:'89950167', 7:'89016252', 8:'90155946', 9:'99999828',
                  10:'99999826', 11:'99999827', 12:'89016259', 13:'99999830', 14:'89016253'}
predict=pd.read_csv('../data/predict.csv',header=None,names=['current_service'])
predict['current_service'] = predict['current_service'].map(label_vocabulary)

id=pd.read_csv('../data/preprocessed_test.csv')['user_id']
result=pd.concat([id,predict],axis=1)
result.to_csv('../data/result.csv',index=False)