import numpy
import nltk

from imblearn.over_sampling import SMOTE
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from code.classification.doc_to_vec import DocToVec, default_model_path
from .preprocess.preprocess import preprocess


stemmer =SnowballStemmer("english")


def create_model_from_training_data(algo, training_data, vector_method, smote=True):
    training_comments = []
    training_ratings = []
    print('Training model...')
    for sentiment_data in training_data:
        comments = preprocess(sentiment_data.text)
        training_comments.append(comments)
        training_ratings.append(sentiment_data.rating)

    vectorizer = None
    vec_model = None
    if vector_method == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, sublinear_tf=True, max_df=0.5,
                                     stop_words=[], min_df=3)
        X_train = vectorizer.fit_transform(training_comments).toarray()
    elif vector_method == 'doc2vec':
        vec_model = DocToVec(model=default_model_path)
        X_train = vec_model.get_doc_to_vec_array(training_comments)

    Y_train = numpy.array(training_ratings)

    # Apply SMOTE to improve ratio of the minority class
    if smote:
        smote_model = SMOTE(ratio='auto', random_state=None, k=None, k_neighbors=5, m=None, m_neighbors=15, out_step=.0001,
                            kind='regular', svm_estimator=None, n_jobs=1)
        X_train, Y_train = smote_model.fit_sample(X_train, Y_train)

    # train model
    model = get_classifier(algo)
    model.fit(X_train, Y_train)

    model_vectorizer = {
        'model': model,
        'vectorizer': vectorizer,
    }

    print('Training model complete.')
    return model_vectorizer, vec_model


def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(stemmer.stem(item))
    return stems


def get_classifier(algo):
    if algo=="GBT":
        return GradientBoostingClassifier()
    elif algo=="RF":
        return  RandomForestClassifier()
    elif algo=="ADB":
        return AdaBoostClassifier()
    elif algo =="DT":
        return  DecisionTreeClassifier()
    elif algo=="NB":
        return  BernoulliNB(alpha=0.5)
    elif algo == 'MulNB':
        return MultinomialNB()
    elif algo == 'GNB':
        return GaussianNB()
    elif algo=="SGD":
        return  SGDClassifier()
    elif algo=="SVC":
        return LinearSVC()
    elif algo=="MLPC":
        return MLPClassifier(activation='logistic',  batch_size='auto',
        early_stopping=True, hidden_layer_sizes=(100,), learning_rate='adaptive',
        learning_rate_init=0.1, max_iter=5000, random_state=1,
        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
        warm_start=False)
    elif algo == 'bagging':
        return BaggingClassifier()
    elif algo == 'extratree':
        return ExtraTreesClassifier()
    elif algo == 'voting':
        return VotingClassifier(estimators=[('lr', LogisticRegression(random_state=1)), ('rf', RandomForestClassifier(random_state=1)), ('gnb', GaussianNB())])
    elif algo == 'treeExtra':
        return ExtraTreeClassifier()
    elif algo == 'ridge':
        return RidgeClassifier()
    elif algo == 'ridgeCV':
        return RidgeClassifierCV()
    elif algo == 'PAC':
        return PassiveAggressiveClassifier()
    return 0