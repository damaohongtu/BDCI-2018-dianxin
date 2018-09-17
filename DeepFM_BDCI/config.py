
# set the path-to-files
TRAIN_FILE = "data/preprocessed_train.csv"
TEST_FILE = "data/preprocessed_test.csv"

SUB_DIR = "./output"


NUM_SPLITS = 5
RANDOM_SEED = 42

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
          "service_type",
          "is_mix_service",
          "many_over_bill",
          "contract_type",
          "is_promise_low_consume"
          "net_service",
          "gender",
          "complaint_level",
]

NUMERIC_COLS = [
    # binary

    # numeric
     "age",
     "online_time",
     "1_total_fee",
     "2_total_fee",
     "3_total_fee",
     "4_total_fee",
     "month_traffic",
     "contract_time",
     "pay_times",
     "pay_num",
     "last_month_traffic",
     "local_trafffic_month",
     "local_caller_time",
     "service1_caller_time",
     "service2_caller_time",
     "former_complaint_num",
     "former_complaint_fee",
    # feature engineering
]

IGNORE_COLS = [
]