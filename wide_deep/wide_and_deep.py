#!/usr/bin/env python
# coding=utf-8
import tempfile
import tensorflow as tf

from six.moves import urllib

import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir","","Base directory for output models.")
flags.DEFINE_string("model_type","wide_n_deep","valid model types:{'wide','deep', 'wide_n_deep'")
flags.DEFINE_integer("train_steps",200,"Number of training steps.")
flags.DEFINE_string("train_data","../data/train.csv", "Path to the training data.")
flags.DEFINE_string("test_data", "../data/eval.csv", "path to the test data")

COLUMNS = ["service_type","is_mix_service","many_over_bill","contract_type","is_promise_low_consume",
           "net_service","gender","complaint_level","age","online_time","1_total_fee","2_total_fee",
           "3_total_fee","4_total_fee","month_traffic", "contract_time","pay_times","pay_num",
           "last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time",
           "service2_caller_time","former_complaint_num","former_complaint_fee"]

LABEL_COLUMN = "current_service"

CATEGORICAL_COLUMNS = ["service_type","is_mix_service","many_over_bill","contract_type",
                       "is_promise_low_consume","net_service","gender","complaint_level"]

CONTINUOUS_COLUMNS = ["age","online_time","1_total_fee","2_total_fee","3_total_fee","4_total_fee","month_traffic",
                      "contract_time","pay_times","pay_num","last_month_traffic","local_trafffic_month",
                      "local_caller_time","service1_caller_time","service2_caller_time","former_complaint_num","former_complaint_fee"]

# build the estimator
def build_estimator(model_dir):
    # 离散分类别的
    service_type=tf.contrib.layers.sparse_column_with_keys(column_name="service_type",keys=[4,1,3])
    is_mix_service=tf.contrib.layers.sparse_column_with_keys(column_name="is_mix_service", keys=[0,1])
    many_over_bill=tf.contrib.layers.sparse_column_with_keys(column_name="is_over_bill", keys=[0,1])
    contract_type=tf.contrib.layers.sparse_column_with_keys(column_name="contract_type", keys=[1,0,3,9,2,12,6,7,8])
    is_promise_low_consume=tf.contrib.layers.sparse_column_with_keys(column_name="is_promise_low_consume", keys=[])
    net_service=tf.contrib.layers.sparse_column_with_keys(column_name="net_service", keys=[4,2,3,9])
    gender=tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=[0,1,2])
    complaint_level=tf.contrib.layers.sparse_column_with_keys(column_name="complaint_level", keys=[0,1,2,3])

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    online_time=tf.contrib.layers.real_valued_column("online_time")
    total_fee_1=tf.contrib.layers.real_valued_column("1_total_fee")
    total_fee_2=tf.contrib.layers.real_valued_column("2_total_fee")
    total_fee_3=tf.contrib.layers.real_valued_column("3_total_fee")
    total_fee_4=tf.contrib.layers.real_valued_column("4_total_fee")
    month_traffic=tf.contrib.layers.real_valued_column("month_traffic")
    contract_time=tf.contrib.layers.real_valued_column("contract_time")
    pay_times=tf.contrib.layers.real_valued_column("pay_times")
    pay_num=tf.contrib.layers.real_valued_column("pay_num")
    last_month_traffic=tf.contrib.layers.real_valued_column("last_month_traffic")
    local_trafffic_month=tf.contrib.layers.real_valued_column("local_trafffic_month")
    local_caller_time=tf.contrib.layers.real_valued_column("local_caller_time")
    service1_caller_time=tf.contrib.layers.real_valued_column("service1_caller_time")
    service2_caller_time=tf.contrib.layers.real_valued_column("service2_caller_time")
    former_complaint_num=tf.contrib.layers.real_valued_column("former_complaint_num")
    former_complaint_fee=tf.contrib.layers.real_valued_column("former_complaint_fee")

    #类别转换
    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries= [18,25, 30, 35, 40, 45, 50, 55, 60, 65])

    wide_columns = [service_type,is_mix_service,many_over_bill,contract_type,
                    is_promise_low_consume,net_service,gender,complaint_level,age_buckets,
                    tf.contrib.layers.crossed_column([age_buckets], hash_bucket_size=int(1e6)),
                    ]

    #embedding_column用来表示类别型的变量
    deep_columns = [tf.contrib.layers.embedding_column(gender, dimension=8),
                    age,online_time,total_fee_1,total_fee_2,total_fee_3,total_fee_4,month_traffic,
                    contract_time,pay_times,pay_num,last_month_traffic,local_trafffic_month,
                    local_caller_time,service1_caller_time,service2_caller_time,former_complaint_num,former_complaint_fee]

    if FLAGS.model_type =="wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100,50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns, dnn_feature_columns = deep_columns, dnn_hidden_units=[100,50])

    return m

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(indices=[[i,0] for i in range( df[k].size)], values = df[k].values, shape=[df[k].size,1]) for k in CATEGORICAL_COLUMNS}#原文例子为dense_shape
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

train_file_name="../data/train.csv"
test_file_name="../data/eval.csv"

def train_and_eval():
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python"
    )
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python"
    )

    # drop Not a number elements
    df_train = df_train.dropna(how='any',axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    #convert >50 to 1
    # df_train[LABEL_COLUMN] = (
    #     df_train["income_bracket"].apply(lambda x: ">50" in x).astype(int)
    # )
    # df_test[LABEL_COLUMN] = (
    #     df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model dir = %s" % model_dir)

    m = build_estimator(model_dir)
    print (FLAGS.train_steps)
    m.fit(input_fn=lambda: input_fn(df_train),
          steps=FLAGS.train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

    for key in sorted(results):
        print("%s: %s"%(key, results[key]))

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()