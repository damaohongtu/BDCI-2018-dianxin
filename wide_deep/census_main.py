"""Train DNN on census income dataset."""

import os

from absl import app as absl_app
from absl import flags
import tensorflow as tf

from utils.flags import core as flags_core
from utils.logs import logger
from wide_deep import census_dataset
from wide_deep import wide_deep_run_loop
from tensorflow.python.ops.losses import losses

def define_census_flags():
  wide_deep_run_loop.define_wide_deep_flags()
  flags.adopt_module_key_flags(wide_deep_run_loop)
  flags_core.set_defaults(data_dir='../data',
                          model_dir='./census_model',
                          train_epochs=40,
                          epochs_between_evals=1,
                          inter_op_parallelism_threads=0,
                          intra_op_parallelism_threads=0,
                          batch_size=40)


def build_estimator(model_dir, model_type, model_column_fn, inter_op, intra_op):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = model_column_fn()
  hidden_units = [100, 75, 50, 25]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 1},
                                    inter_op_parallelism_threads=inter_op,
                                    intra_op_parallelism_threads=intra_op))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        n_classes=15,
        label_vocabulary=['99999825', '90063345', '90109916', '89950166', '89950168',
                          '99104722', '89950167', '89016252', '90155946', '99999828',
                          '99999826', '99999827', '89016259', '99999830', '89016253'])


def run_census(flags_obj):
  train_file = os.path.join(flags_obj.data_dir, census_dataset.TRAINING_FILE)
  test_file = os.path.join(flags_obj.data_dir, census_dataset.EVAL_FILE)

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return census_dataset.input_fn(
        data_file=train_file,
        num_epochs=flags_obj.epochs_between_evals,
        shuffle=True,
        batch_size=flags_obj.batch_size)

  def eval_input_fn():
    return census_dataset.input_fn(
        data_file=test_file,
        num_epochs=1,
        shuffle=False,
        batch_size=flags_obj.batch_size)

  tensors_to_log = {
      'average_loss': '{loss_prefix}head/truediv',
      'loss': '{loss_prefix}head/weighted_loss/Sum'
  }

  wide_deep_run_loop.run_loop(
      name="面向电信行业存量用户的智能套餐个性化匹配模型",
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      model_column_fn=census_dataset.build_model_columns,
      build_estimator_fn=build_estimator,
      flags_obj=flags_obj,
      tensors_to_log=tensors_to_log,
      early_stop=True
  )


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_census(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_census_flags()
  absl_app.run(main)
