import pandas as pd
from sklearn.model_selection import train_test_split

train = pd.read_csv('../data/preprocessed_train.csv', encoding='utf-8')
test = pd.read_csv('../data/preprocessed_test.csv', encoding='utf-8')

train_df, eval_df = train_test_split(train, test_size=0.2, shuffle=True)
train_df.to_csv('../data/train.csv', index=False, encoding='utf-8')
eval_df.to_csv('../data/eval.csv', index=False, encoding='utf-8')
