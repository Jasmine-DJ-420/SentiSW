# this class introduces how to use our tool.
import argparse
from code.classification.classifier import Classifier
from code.entity.training_set_generation import get_entity

parser = argparse.ArgumentParser()

def get_tuple(text):
    sentiment_analyzer = Classifier(read=True, vector_method='tfidf')
    # sentiment_analyzer.save_model()
    sentiment = sentiment_analyzer.get_sentiment_polarity(text)[0]
    ret = {'sentiment': sentiment}
    if sentiment != 'Neutral':
        entity = get_entity(text)
        ret['entity'] = entity
    return ret


if __name__ == '__main__':
    parser.add_argument('--text', required=True, help='issue text contents')

    parser.print_help()

    text = parser.parse_args().text

    print(get_tuple(text))
