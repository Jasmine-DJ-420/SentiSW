# 提取词干, rule-based
from code.classification.preprocess.preprocess import preprocess_v2
from code.tools import csv_op
from settings import dir_path
from nltk import word_tokenize, pos_tag
from nltk.tag.stanford import StanfordNERTagger
from nltk.stem.snowball import SnowballStemmer
import csv

feature_list = None
st = None
stemmer = SnowballStemmer("english")
word_tag_header = ['word_tag', 'feature_index', '_id', '_sen_id']
pos_word_tag_path = 'pos_word_tag.csv'
neg_word_tag_path = 'neg_word_tag.csv'
stanford_nlp_model = dir_path + '/lib/stanford_nlp/english.all.3class.distsim.crf.ser.gz'
stanford_nlp_jar = dir_path + '/lib/stanford_nlp/stanford-ner.jar'
senti_features_dict = dir_path + '/data/feature_importance.csv'


def get_stanford_ner():
    global st
    if st is None:
        st = StanfordNERTagger(
            model_filename=stanford_nlp_model,
            path_to_jar=stanford_nlp_jar,
        )
    return st


header =['feature_names', 'importance']
def read_feature():
    global feature_list
    if feature_list is None:
        feature_list = []
        with open(senti_features_dict, 'r') as f:
            reader = csv.DictReader(f)
            for line in reader:
                if line[header[1]] != '0.0':
                    feature_list.append(line[header[0]])
    return feature_list


# 识别出
def get_sentences(text):
    sentences = preprocess_v2(text)
    # sentences = sent_tokenize(text)
    ret = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        ret.append({
            'word_tag': combine_pos_ner(words),
            'feature_index': get_feature_index(words),
        })
    return ret


# 识别一句话feature的最小index
def get_feature_index(words):
    feature_list = read_feature()
    feature_index = -1
    for word in words:
        word = stem_word(word)
        if word not in feature_list:
            continue
        index = feature_list.index(word)
        feature_index = min(feature_index, index) if feature_index != -1 else index
    return feature_index

# NER
def get_ner(words):
    # stanford ner
    st = get_stanford_ner()
    iob_tagged = st.tag(words)
    return iob_tagged


# POS
def get_pos(words):
    return pos_tag(words)


def combine_pos_ner(words):
    ner = get_ner(words)
    part_of_speech = get_pos(words)
    ret = []
    for index in range(len(ner)):
        word_ner = ner[index][0]
        word_pos = part_of_speech[index][0]
        if word_ner != word_pos:
            print('unmatched words.')
        ret.append({
            'word': word_ner,
            'ner': ner[index][1],
            'pos': part_of_speech[index][1],
        })
    return ret


def stem_word(word):
    return stemmer.stem(word)

def save_word_tag(sub_list, path):
    op = csv_op.CsvOp(word_tag_header, path)
    op.init_csv()
    for p in sub_list:
        tag_dict_array = get_sentences(p['body'])
        for index in range(len(tag_dict_array)):
            tag_dict = tag_dict_array[index]
            tag_dict['_id'] = p['_id']
            tag_dict['_sen_id'] = index
            op.write_csv(tag_dict)


def get_entity(text):
    print('Start recognizing entity.')
    sentences = get_sentences(text)
    return recognize_sentences(sentences)

# rule: 距离最近的名词对象是人/ 和PRP相连 VERB + PRP, 考虑@
#       距离最近的名词对象是大写的(可能跟项目有关)
special_stem = ['POS_EMOTICON', 'NEG_EMOTICON', 'EXCLAMATION', 'Capistrano']
def recognize_sentences(sentences):
    sentences = sorted(sentences, key=lambda item:item['feature_index'])
    # 首先只考虑sentence1
    key_sentences = []
    feature_index = 0
    for sentence in sentences:
        if sentence['feature_index'] != '-1':
            if not feature_index or sentence['feature_index'] == feature_index:
                feature_index = sentence['feature_index']
                key_sentences.append(sentence['word_tag'])
            elif int(sentence['feature_index']) > int(feature_index):
                break
    if not key_sentences:
        return

    for key_sentence in key_sentences:
        if get_entity_label_from_sentence(key_sentence) == 'PERSON':
            return 'PERSON'

    return 'PROJECT'


def get_entity_label_from_sentence(key_sentence):
    entity_label = 'PROJECT'
    features = read_feature()
    for index in range(len(key_sentence)):
        word = key_sentence[index]['word']
        ner = key_sentence[index]['ner']
        if ner == 'PERSON' and word not in special_stem or word == '@':
            entity_label = 'PERSON'
            break
        if stem_word(word) == 'not_work':
            entity_label = 'PROJECT'
            break
        if stem_word(word) == 'thank' or stem_word(word) == 'sorri' or\
                stem_word(word) in features \
                and find_PRP(key_sentence, index) \
                and feature_list.index(stem_word(word)) <= 50:
            entity_label = 'PERSON'
            break
    return entity_label


def find_PRP(key_sentence, index, win=1):
    # left
    for index_add in range(1, win+1):
        if index+index_add < len(key_sentence) and (key_sentence[index + index_add]['pos'] == 'PRP'
                                                    or key_sentence[index + index_add]['pos'] == 'PRP$')\
                and key_sentence[index + index_add]['word'].lower() != 'i':
            return True
        if index - index_add >= 0 and (key_sentence[index-index_add]['pos'] == 'PRP'
                                                    or key_sentence[index - index_add]['pos'] == 'PRP$')\
                and key_sentence[index - index_add]['word'].lower() != 'i':
            return True
    return False


if __name__ == '__main__':
    print(get_entity("thank you very much. it's very nice of you."))
