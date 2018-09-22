import numpy as np
import pandas
import os
import pandas as pd
exclude = ['2_total_fee', '3_total_fee', 'age', 'user_id', 'current_service']

class LoadData(object):
    '''
    given the path of data, return the data format for DeepFM
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X' refers to a list of features_M dimension vectors with 0 or 1 entries
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type):
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + "final_train.csv"
        self.testfile = self.path + dataset + "test.csv"
        self.validationfile = self.path + dataset + "dev.csv"

        self.class_dict = {
            99999825: 0, 90063345: 1, 90109916: 2, 89950166: 3, 89950168: 4,
            99104722: 5, 89950167: 6, 89016252: 7, 90155946: 8, 99999828: 9,
            99999826: 10, 99999827: 11, 89016259: 12, 99999830: 13, 89016253: 14
            , }
        self.class_dict_reverse = {
            0: 99999825,    1: 90063345,  2: 90109916,  3: 89950166,  4: 89950168,
            5: 99104722,    6: 89950167,  7: 89016252,  8: 90155946,  9: 99999828,
            10: 99999826,  11: 99999827, 12: 89016259, 13: 99999830, 14: 89016253
        }
        self.features_M = self.map_features()

        self.Train_data, self.Validation_data, self.Test_data = self.construct_data(loss_type)



    def map_features(self):  # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        print("features_M:", len(self.features))

        return len(self.features)

    def read_features(self, file):  # read a feature file
        data = pd.read_csv(file, encoding='utf-8')
        for feature in data.columns.values:
            if feature in ['current_service', 'user_id']: continue
            for index in data.index:
                i = len(self.features)
                item = data[feature].values[index]
                if item not in self.features:
                    self.features[item] = i
                    i += 1


    def construct_data(self, loss_type):
        X_, Y_, user_id = self.read_data(self.trainfile)

        Train_data = self.construct_dataset(X_, Y_)

        print("# of training:", len(Y_))

        X_, Y_, user_id = self.read_data(self.validationfile)

        Validation_data = self.construct_dataset(X_, Y_)


        print("# of validation:", len(Y_))

        X_, user_id = self.read_data(self.testfile)

        Test_data = self.construct_test(X_, user_id)


        print("# of test:", len(Y_))

        return Train_data, Validation_data, Test_data



    def read_data(self, file):
        # maped to indexs in self.features
        def trans_label(key):
            return self.class_dict[key]
        def trans_data(key):
            return self.features[key]
        data = pd.read_csv(file)
        user_id= data['user_id'].values
        data.drop(['user_id'], axis=1, inplace=True)
        if 'current_service' in data.columns.values:
            Y = data['current_service'].apply(trans_label).values

            data.drop(['current_service'], axis=1, inplace=True)
            X = data.values
            X = [[trans_data(key) for key in line] for line in X]
            return X, Y, user_id
        else:
            X = data.values
            X = [[trans_data(key) for key in line] for line in X]
            return X, user_id

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}

        Data_Dic['Y'] = Y_

        Data_Dic['X'] = X_
        return Data_Dic

    def construct_test(self, X_, user_id):
        Data_Dic = {}

        Data_Dic['X'] = X_
        Data_Dic['user_id'] = user_id
        return Data_Dic
