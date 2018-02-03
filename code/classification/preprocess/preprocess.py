# This class is for preprocessing logic
from . import markdown_patterns, emoticons, stop_words
from nltk.stem.snowball import SnowballStemmer
import re, nltk, csv
from settings import dir_path


# define markdown parsing
def markdown_parser(text):
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    code_flag = ''
    after_parser = ''
    for line in text.split('\n'):
        if code_flag:
            if line.startswith(code_flag):
                code_flag = ''
            if code_flag in markdown_patterns.code_n:
                line = line.rstrip()
                if line.startswith(code_flag) or line.endswith(code_flag):
                    code_flag = ''
            continue
        for code_pattern in markdown_patterns.code:
            if line.startswith(code_pattern):
                code_flag = code_pattern
                break
        # for code_pattern in markdown_patterns.code_n:
        #     if line.startswith(code_pattern):
        #         code_flag = code_pattern
        #         break
        if not code_flag:
            if if_quote(line):
                continue
            for inline_pattern in markdown_patterns.inline:
                # line.replace(inline_pattern, '')
                line = re.sub(inline_pattern, '', line)
            line = line.strip()
            if line:
                after_parser += line
                after_parser += '\n'

    return after_parser


# 判断是否是quote句
def if_quote(line):
    for quote in markdown_patterns.quote:
        if line.startswith(quote):
            return True
    return False


# 将所有非ascii码字符转化为空格
def trans_ascii(text):
    after_parsing = ''
    for letter in text:
        after_encode = letter.encode('ascii', 'ignore')
        letter = after_encode if after_encode else b' '
        after_parsing += letter.decode('ascii')
    return after_parsing


negation_list = ['no', 'not', 'none', 'no one', 'nobody', 'nothing', 'neither', 'nowhere', 'never',
                 'hardly', 'scarcely', 'barely', 'little', 'few', 'rarely', 'seldom']
# 处理否定句子
word_regex = r'[^0-9a-zA-Z_]'
word_regex_with_at = r'[^0-9a-zA-Z_@]'
def prepare_not(text, lowercase=False):
    ret = ''
    if lowercase:
        _not = 'not_%s '
        mark_regex = word_regex_with_at

    else:
        _not = 'NOT_%s '
        mark_regex = word_regex
    for sen in nltk.sent_tokenize(text):
        # 处理emoticons
        sen = emoticons.emoticon_parser(sen)
        sen = punctuation_marks(sen)
        sen = re.sub(mark_regex, ' ', sen)
        part_of_speech = nltk.tag.pos_tag(nltk.word_tokenize(sen), tagset='universal')
        neg_count = 0
        for word, pos in part_of_speech:
            if word in negation_list: neg_count += 1
        if neg_count % 2 == 1:
            for word, pos in part_of_speech:
                # 删除无用标点
                if (pos == 'ADV' or pos == 'ADJ' or pos == 'VERB') and word not in negation_list:
                    ret += _not % word
                elif word not in negation_list:
                    ret += '%s ' % word
        else:
            ret += (sen + '\n')
    return ret


# 处理缩写
full_path = dir_path + '/lib/statics/Contractions.txt'
with open(full_path, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    contraction_list = {rows[0]: rows[1].strip() for rows in reader}
def full_expression(text):
    for contraction in contraction_list:
        text = re.sub(contraction, contraction_list[contraction], text)
    return text


# 处理特殊标点
punctuation = {
    r'\.[\.]+': ' ELLIPSIS ',
    r'!+': ' EXCLAMATION ',
    r'\?+': ' QUESTION_MARK ',
}
def punctuation_marks(text):
    for p in punctuation:
        text = re.sub(p, punctuation[p], text)
    return text

# 停止词
def remove_stop_words(text):
    ret = ''
    for word in nltk.word_tokenize(text):
        # ignore upper case
        word = word.lower() if word not in ['POS_EMOTICON', 'NEG_EMOTICON'] else word
        if word not in stop_words.common_stop_words and word not in stop_words.software_stop_words:
            ret += '%s ' % stem(word)
        else:
            ret += ' '
    return ret


# stem
stemmer = SnowballStemmer("english")
def stem(word):
    return stemmer.stem(word)


def stem_text(text):
    ret = ''
    for word in nltk.word_tokenize(text):
        ret += '%s ' % stem(word)

    return ret


# 预处理所有过程
def preprocess(text):
    text = trans_ascii(text)
    text = markdown_parser(text)
    text = full_expression(text)
    text = remove_stop_words(text)
    text = prepare_not(text)
    return text


# 不去除停止词的预处理过程, 返回句子
def preprocess_v2(text):
    text = trans_ascii(text)
    text = markdown_parser(text)
    sentences = []
    for sentence in nltk.sent_tokenize(text):
        sentence = full_expression(sentence)
        sentence = prepare_not(sentence, lowercase=True)
        sentences.append(sentence)
    return sentences


# test preprocessing
if __name__ == '__main__':
    # collection = pymongo.MongoClient("localhost", 27017).github_issue['Annotation']
    # comments = collection.find({}).skip(1362).limit(1)
    # comments = collection.find({'old_id': 128768497})
    comments = [{'text':
        'Okay thats probably the problem, but can you explain what it means. Im no programming wonder, sorry ;)'}]
    for comment in comments:
        print('origin: %s' % comment['text'])
        print('-----------------------')
        print('after_parser: %s' % preprocess(comment['text']))
    # comment = "i don't know what's going on here. but i feel liking it."
    # comment = preprocess(comment)
    # print(comment)
    # print(preprocess(comment))
