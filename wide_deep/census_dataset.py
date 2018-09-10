"""Download and clean the Census Income Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

TRAINING_FILE = 'train.csv'
EVAL_FILE = 'test.csv'



_CSV_COLUMNS = ["service_type","is_mix_service","online_time","1_total_fee","2_total_fee",
           "3_total_fee","4_total_fee","month_traffic","many_over_bill","contract_type",
           "contract_time","is_promise_low_consume","net_service","pay_times","pay_num",
           "last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time",
           "service2_caller_time","gender","age","complaint_level","former_complaint_num",
           "former_complaint_fee","current_service","user_id"]

_CSV_COLUMN_DEFAULTS = [[''],[''],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[''],[''],[0],[''], [''],[0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[''],[0],[''],[0],[0.0],[''],['']]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}


def build_model_columns():
  """Builds a set of wide and deep feature columns."""

  # Continuous variable columns
  age = tf.feature_column.numeric_column("age")
  online_time = tf.feature_column.numeric_column("online_time")
  total_fee_1 = tf.feature_column.numeric_column("1_total_fee")
  total_fee_2 = tf.feature_column.numeric_column("2_total_fee")
  total_fee_3 = tf.feature_column.numeric_column("3_total_fee")
  total_fee_4 = tf.feature_column.numeric_column("4_total_fee")
  month_traffic = tf.feature_column.numeric_column("month_traffic")
  contract_time = tf.feature_column.numeric_column("contract_time")
  pay_times = tf.feature_column.numeric_column("pay_times")
  pay_num = tf.feature_column.numeric_column("pay_num")
  last_month_traffic = tf.feature_column.numeric_column("last_month_traffic")
  local_trafffic_month = tf.feature_column.numeric_column("local_trafffic_month")
  local_caller_time = tf.feature_column.numeric_column("local_caller_time")
  service1_caller_time = tf.feature_column.numeric_column("service1_caller_time")
  service2_caller_time = tf.feature_column.numeric_column("service2_caller_time")
  former_complaint_num = tf.feature_column.numeric_column("former_complaint_num")
  former_complaint_fee = tf.feature_column.numeric_column("former_complaint_fee")


  service_type = tf.feature_column.categorical_column_with_vocabulary_list("service_type", ['4', '1', '3'])
  is_mix_service = tf.feature_column.categorical_column_with_vocabulary_list("is_mix_service", ['0', '1'])
  many_over_bill = tf.feature_column.categorical_column_with_vocabulary_list("many_over_bill", ['0', '1'])
  contract_type = tf.feature_column.categorical_column_with_vocabulary_list("contract_type",['1', '0', '3', '9', '2', '12', '6', '7', '8'])
  is_promise_low_consume = tf.feature_column.categorical_column_with_vocabulary_list("is_promise_low_consume",['0', '1'])
  net_service = tf.feature_column.categorical_column_with_vocabulary_list("net_service", ['4', '2', '3', '9'])
  gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ['0', '1', '2'])
  complaint_level = tf.feature_column.categorical_column_with_vocabulary_list("complaint_level",['0', '1', '2', '3'])



  # To show an example of hashing:
  #service_type = tf.feature_column.categorical_column_with_hash_bucket('service_type', hash_bucket_size=_HASH_BUCKET_SIZE)

  # Transformations.
  age_buckets = tf.feature_column.bucketized_column(
      age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

  # Wide columns and deep columns.
  base_columns = [service_type, is_mix_service, many_over_bill, contract_type,
                  is_promise_low_consume,net_service, gender, complaint_level, age_buckets]

  crossed_columns = [
      tf.feature_column.crossed_column(
          ['service_type', 'is_mix_service'], hash_bucket_size=_HASH_BUCKET_SIZE),
      tf.feature_column.crossed_column(
          [age_buckets, 'gender', 'complaint_level'],
          hash_bucket_size=_HASH_BUCKET_SIZE),
  ]

  wide_columns = base_columns + crossed_columns

  deep_columns = [age,online_time,total_fee_1,total_fee_2,total_fee_3,total_fee_4,month_traffic,
                  contract_time,pay_times,pay_num,last_month_traffic,local_trafffic_month,
                  local_caller_time,service1_caller_time,service2_caller_time,former_complaint_num,former_complaint_fee]

  return wide_columns, deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size):
  def parse_csv(value):
    tf.logging.info('Parsing {}'.format(data_file))
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    labels = features.pop('current_service')

    classes = tf.equal(labels, '>50K')  # binary classification

    return features, classes

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset