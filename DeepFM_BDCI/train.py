
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import config
from DataReader import FeatureDictionary, DataParser
from deepfm import DeepFM

def _load_data():

    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    # 对标签编码 映射关系
    label2current_service = dict(
        zip(range(0, len(set(dfTrain['current_service']))), sorted(list(set(dfTrain['current_service'])))))
    current_service2label = dict(
        zip(sorted(list(set(dfTrain['current_service']))), range(0, len(set(dfTrain['current_service'])))))

    dfTrain['current_service'] = dfTrain['current_service'].map(current_service2label)
    print(dfTrain['current_service'].unique())

    cols = [c for c in dfTrain.columns if c not in ["user_id", "current_service"]]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain["current_service"].values
    X_test = dfTest[cols].values
    ids_test = dfTest["user_id"].values
    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices, label2current_service

def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, label2current_service):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    xx_score = []
    cv_pred = []
    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]

    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)

        xx_pred = dfm.predict(Xi_valid_, Xv_valid_)
        xx_score.append(f1_score(y_valid_, xx_pred, average='macro'))

        y_test = dfm.predict(Xi_test, Xv_test)
        if i == 0:
            cv_pred = np.asarray(y_test).reshape(-1, 1)
        else:
            cv_pred = np.hstack((cv_pred, np.asarray(y_test).reshape(-1, 1)))

    submit = []
    for line in cv_pred:
        submit.append(np.argmax(np.bincount(line)))

    # 保存结果
    df_test = pd.DataFrame()
    df_test['id'] = list(ids_test)
    df_test['predict'] = submit
    df_test['predict'] = df_test['predict'].map(label2current_service)

    df_test.to_csv('result.csv', index=False)

    print(xx_score, np.mean(xx_score))

    return y_train_meta, y_test_meta




# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices, label2current_service = _load_data()

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))


# ------------------ DeepFM Model ------------------
# params
dfm_params = {
    "embedding_size": 16,
    "dropout_fm": [1.0, 1.0],
    "deep_layers": [64, 64],
    "dropout_deep": [1.0, 1.0, 1.0],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 1000,
    "batch_size": 1024,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.000001,
    "verbose": True,
    "random_seed": config.RANDOM_SEED
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params, label2current_service)




