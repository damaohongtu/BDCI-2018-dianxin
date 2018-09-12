import pandas as pd

train_file = './input/train.csv'
test_file = './input/test.csv'

def normalization():
    train = pd.read_csv(train_file, encoding='utf-8')
    test = pd.read_csv(train_file, encoding='utf-8')
    print(train.columns)

    # 对标签编码 映射关系
    label2current_service = dict(
        zip(range(0, len(set(train['current_service']))), sorted(list(set(train['current_service'])))))
    current_service2label = dict(
        zip(sorted(list(set(train['current_service']))), range(0, len(set(train['current_service'])))))

    # 原始数据的标签映射
    train['current_service'] = train['current_service'].map(current_service2label)
    train.to_csv('./data/train.csv', index=False)
    test.to_csv('./data/test.csv', index=False)
def main():
    normalization()

if __name__ == '__main__':
    main()