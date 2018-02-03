from sklearn.model_selection import KFold
from code.classification.classifier import Classifier
from code.classification.file import get_training_data
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from statistics import mean
import random
import numpy
import csv
from lib.statics.classification_lists import classification_models

import os

dir_path = 'validation_result/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

def ten_fold_cross_validation(dataset, ALGO):
    kf = KFold(n_splits=10, shuffle=True)

    run_precision = []
    run_recall = []
    run_f1score = []
    run_accuracy = []

    count=1

    #Randomly divide the dataset into 10 partitions
    # During each iteration one partition is used for test and remaining 9 are used for training
    for train, test in kf.split(dataset):
        print("Using split-"+str(count)+" as test data..")
        classifier_model = Classifier(algo=ALGO, training_data=dataset[train], vector_method='tfidf')

        test_comments=[comments.text for comments in dataset[test]]
        test_ratings=[comments.rating for comments in dataset[test]]

        pred = classifier_model.get_sentiment_polarity_collection(test_comments)

        label_list = ['Negative', 'Positive', 'Neutral']
        precision = precision_score(test_ratings, pred, labels=label_list, average=None)
        recall = recall_score(test_ratings, pred, labels=label_list, average=None)
        f1score = f1_score(test_ratings, pred, labels=label_list, average=None)
        accuracy = accuracy_score(test_ratings, pred)

        run_accuracy.append(accuracy)
        run_f1score.append(f1score)
        run_precision.append(precision)
        run_recall.append(recall)
        count+=1

    return (get_mean_list(run_precision),get_mean_list(run_recall),get_mean_list(run_f1score),mean(run_accuracy))


def get_mean_list(run_result):
    label_list = {}
    label_list['Negative'] = mean([p[0] for p in run_result])
    label_list['Positive'] = mean([p[1] for p in run_result])
    label_list['Neutral'] = mean(p[2] for p in run_result)
    return label_list


def get_mean_dict(run_result):
    label_list = {}
    label_list['Negative'] = mean([p['Negative'] for p in run_result])
    label_list['Positive'] = mean([p['Positive'] for p in run_result])
    label_list['Neutral'] = mean([p['Neutral'] for p in run_result])
    return label_list


def validation_list(algo):
    ALGO = algo
    REPEAT = 1

    print("Cross validation")
    print("Algrithm: " + ALGO)
    print("Repeat: " + str(REPEAT))

    training_data = get_training_data()
    random.shuffle(training_data)
    training_data = numpy.array(training_data)

    Precision = []
    Recall = []
    Fmean = []
    Accuracy = []

    for k in range(0, REPEAT):
        print(".............................")
        print("Run# {}".format(k))
        (precision, recall, f1score, accuracy) = ten_fold_cross_validation(training_data, ALGO)
        Precision.append(precision)
        Recall.append(recall)
        Fmean.append(f1score)
        Accuracy.append(accuracy)
        print("Precision: %s" % precision)
        print("Recall: %s" % recall)
        print("F-measure: %s" % f1score)
        print("Accuracy: %s" % accuracy)

    with open("%scross-validation-%s_100.csv" % (dir_path, ALGO), 'w') as file:
        header = ['Run', 'Precision', 'Recall', 'Fscore', 'Accuracy']
        writer = csv.DictWriter(file, header)
        writer.writeheader()
        for k in range(0, REPEAT):
            row = {'Run': k, 'Precision': Precision[k], 'Recall': Recall[k], 'Fscore': Fmean[k],
                   'Accuracy': Accuracy[k]}
            writer.writerow(row)
        row = {'Run': 'Average', 'Precision': get_mean_dict(Precision), 'Recall': get_mean_dict(Recall),
               'Fscore': get_mean_dict(Fmean), 'Accuracy': mean(Accuracy)}
        writer.writerow(row)

    print("-------------------------")
    print("Average Precision: %s" % (get_mean_dict(Precision)))
    print("Average Recall: %s" % get_mean_dict(Recall))
    print("Average Fmean: %s" % get_mean_dict(Fmean))
    print("Average Accuracy: %s" % (mean(Accuracy)))
    print("-------------------------")


if __name__ == '__main__':
    for algo in classification_models:
        validation_list(algo)
