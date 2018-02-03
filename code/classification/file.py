# coding:utf-8
from code.classification.SentimentData import SentimentData
import csv
from settings import dir_path

training_path = dir_path + '/data/training_set_3000.csv'

def get_training_data(training_path=training_path):
    training_data = []
    print('Read training data...')
    with open(training_path, 'r', encoding='utf8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            sentimentdata = SentimentData(row['text'], row['Annotation'])
            training_data.append(sentimentdata)

    # ## print training data
    # for row in training_data:
    #     print(row.text, row.rating)
    return training_data


if __name__ == '__main__':
    get_training_data()