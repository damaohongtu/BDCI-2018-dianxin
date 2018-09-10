import tensorflow as tf

_CSV_COLUMNS = [
        'service_type', 'is_mix_service', 'online_time', '1_total_fee',
        '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
        'many_over_bill', 'contract_type', 'contract_time',
        'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num',
        'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
        'service1_caller_time', 'service2_caller_time', 'gender', 'age',
        'complaint_level', 'former_complaint_num', 'former_complaint_fee',
        'current_service', 'user_id'
]

_CSV_COLUMN_DEFAULTS = [
            [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0],
            [0], ['']
]

_HASH_BUCKET_SIZE = 1000

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    # Continuous variable columns
    age = tf.feature_column.numeric_column('age')

    total_fee_1 = tf.feature_column.numeric_column('1_total_fee')
    total_fee_2 = tf.feature_column.numeric_column('2_total_fee')
    total_fee_3 = tf.feature_column.numeric_column('3_total_fee')
    total_fee_4 = tf.feature_column.numeric_column('4_total_fee')
    online_time = tf.feature_column.numeric_column('online_time')
    month_traffic = tf.feature_column.numeric_column('month_traffic')
    contract_time = tf.feature_column.numeric_column('contract_time')
    pay_times = tf.feature_column.numeric_column('pay_times')
    pay_num = tf.feature_column.numeric_column('pay_num')
    last_month_traffic = tf.feature_column.numeric_column('last_month_traffic')
    local_trafffic_month = tf.feature_column.numeric_column('local_trafffic_month')
    local_caller_time = tf.feature_column.numeric_column('local_caller_time')
    service1_caller_time = tf.feature_column.numeric_column('service1_caller_time')
    service2_caller_time = tf.feature_column.numeric_column('service2_caller_time')
    former_complaint_num = tf.feature_column.numeric_column('former_complaint_num')
    former_complaint_fee = tf.feature_column.numeric_column('former_complaint_fee')


    age_bucket = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    )

    service_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'service_type', [1, 3, 4])
    is_mix_service = tf.feature_column.categorical_column_with_vocabulary_list(
        'is_mix_service', [0, 1])
    many_over_bill = tf.feature_column.categorical_column_with_vocabulary_list(
        'many_over_bill', [0, 1]
    )
    contract_type = tf.feature_column.categorical_column_with_vocabulary_list(
        'contract_type', [1,  0,  3,  9,  2, 12,  6,  7,  8]
    )
    is_promise_low_consume = tf.feature_column.categorical_column_with_vocabulary_list(
        'is_promise_low_consume', [0, 1]
    )
    net_service = tf.feature_column.categorical_column_with_vocabulary_list(
        'net_service', [4, 2, 3, 9]
    )
    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        'gender', [0, 1, 2]
    )
    complaint_level = tf.feature_column.categorical_column_with_vocabulary_list(
        'complaint_level', [0, 2, 1, 3]
    )


    # Wide columns 基本上都是 categorical 特征
    base_columns = [
        service_type, is_mix_service, many_over_bill, contract_type, is_promise_low_consume,
        net_service, gender, complaint_level
    ]
    # 需要交叉的特征
    # 不懂这个hash_backet_size 是神马意思
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['gender', age_bucket], hash_bucket_size=_HASH_BUCKET_SIZE
        )
    ]

    wide_columns = base_columns + crossed_columns

    # 一般为连续特征，样例中加入了一些one_hot编码或者embedding的种类特征
    deep_columns = [
            age,
            total_fee_1,
            total_fee_2,
            total_fee_3,
            total_fee_4,
            online_time,
            month_traffic,
            contract_time,
            pay_times,
            pay_num,
            last_month_traffic,
            local_trafffic_month,
            local_caller_time,
            service1_caller_time,
            service2_caller_time,
            former_complaint_num,
            former_complaint_fee,
    ]

    return wide_columns, deep_columns


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), (
        '%s not found. Please make sure you have '
        'set the --data_dir argument to the correct path.' % data_file)
    def parse_csv(value):
        def parse_csv(value):
            tf.logging.info('Parsing {}'.format(data_file))
            columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
            features = dict(zip(_CSV_COLUMNS, columns))