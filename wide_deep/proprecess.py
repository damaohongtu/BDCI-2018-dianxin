import pandas as pd

train_file = './input/train.csv'
test_file = './input/test.csv'

def normalization():
    train = pd.read_csv(train_file,encoding='utf-8')
    test = pd.read_csv(train_file,encoding='utf-8')
    print(train.columns)



def main():
    normalization()

if __name__ == '__main__':
    main()