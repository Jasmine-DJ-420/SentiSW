import gensim
import numpy as np
import nltk
from gensim.models.doc2vec import TaggedDocument
from code.classification.preprocess.preprocess import preprocess
import pymongo
from settings import dir_path

stem = nltk.stem.SnowballStemmer("english")
default_model_path = dir_path + '/data/model/doc2vec/doc_2_vec.model'


class DocToVec:
    def __init__(self, model=None, training_list=None):
        print('Init doc2vec model.')
        if not model:
            if training_list:
                self.training_list = training_list
            else:
                self.training_list = self.read_data_all()
            self.model = self.train_doc2vec_model(self.training_list)
        else:
            self.model = gensim.models.doc2vec.Doc2Vec.load(model)


    # 训练doc2vec模型
    def train_doc2vec_model(self, training_list):
        tag_doc_list = []
        for index in range(len(training_list)):
            word_list = self.get_unicode_string_list(training_list[index])
            tag_doc_list.append(TaggedDocument(word_list, [str(index)]))
        model = gensim.models.Doc2Vec(tag_doc_list, size=50, window=8, min_count=10, workers=8)
        return model

    # 存储模型
    def save_model(self):
        self.model.save(default_model_path)
        print('Saved successfully')

    # 获取文本向量
    def get_doc_to_vec(self, comments):
        word_list = self.get_unicode_string_list(comments)
        return self.model.infer_vector(word_list).reshape(1, -1)

    # doc2vec training array，返回一个numpy array
    def get_doc_to_vec_array(self, training_comments=None):
        if training_comments:
            vecs = []
            for comment in training_comments:
                word_list = self.get_unicode_string_list(comment)
                vecs.append(self.model.infer_vector(word_list))
            return np.array(vecs)
        return np.array(self.model.docvecs)

    # 分词转unicode
    def get_unicode_string_list(self, text):
        unicode_string_list = []
        tokens = nltk.word_tokenize(text)
        for item in tokens:
            unicode_string_list.append(stem.stem(item))
        return unicode_string_list

    # 读取存储的所有
    def read_data_all(self):
        database = pymongo.MongoClient('127.0.0.1', 27017).github_issue
        body = []
        for issue in database['Issue'].find({}):
            body.append(preprocess(issue['body']))
        for comment in database['IssueComment'].find({}):
            body.append(preprocess(comment['body']))
        print('len of training set: %s' % len(body))
        return body


# 3000训练集
def read_training_set():
    database = pymongo.MongoClient('127.0.0.1', 27017).github_issue
    body = []
    for issue in database['Annotation'].find({}):
        body.append(preprocess(issue['text']))
    print('len of training set: %s' % len(body))
    return body


if __name__ == '__main__':
    doctovec = DocToVec()
    doctovec.save_model()