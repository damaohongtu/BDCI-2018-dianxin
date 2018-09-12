import tensorflow as tf
import os
import numpy as np
import census_dataset
import census_main
exported_path = './census_model'
predictionoutputfile = '../data/test_output.csv'
predictioninputfile = '../data/test.csv'

_CSV_COLUMNS = ["service_type","is_mix_service","online_time","1_total_fee","2_total_fee",
                "3_total_fee","4_total_fee","month_traffic","many_over_bill","contract_type",
                "contract_time","is_promise_low_consume","net_service","pay_times","pay_num",
                "last_month_traffic","local_trafffic_month","local_caller_time","service1_caller_time",
                "service2_caller_time","gender","age","complaint_level","former_complaint_num",
                "former_complaint_fee","user_id"]

_CSV_COLUMN_DEFAULTS = [[''],[''],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[''],[''],[0],[''], [''],[0],[0.0],[0.0],[0.0],[0.0],[0.0],[0.0],[''],[0],[''],[0.0],[0.0],['']]

def input_fn(filenames, num_epochs, batch_size=1):
    def parse_csv(line):
        print('Parsing', filenames)
        columns = tf.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        return features

    # extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(filenames) # can pass one filename or filename list

    # multi-thread pre-process then prefetch
    dataset = dataset.map(parse_csv, num_parallel_calls=10).prefetch(500000)

    # call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features= iterator.get_next()

    return features



def main(_):
    test='../data/test.csv'
    wide_n_deep=census_main.build_estimator("./census_model", 'wide_deep',census_dataset.build_model_columns,0, 0)
    predict=wide_n_deep.predict(input_fn=lambda:input_fn(test, num_epochs=1, batch_size=128),predict_keys="probabilities")
    with open('../data/predict.csv','w')as fo:
        for prob in predict:
            fo.write(str(np.argmax(prob['probabilities']))+'\n')


if __name__ == "__main__":
    tf.app.run()