# 识别emoticons

import re, csv

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=8xX%B]'
Wink = r'[;]'

NoseArea = r'(| |o|O|-|\'|\'-|,|\^|_)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]\}3]'
SadMouths = r'[\(\[</\\cCO0#&EFSsL\|]'
Tongue = r'[pPb9]'
OtherMouths = r'[doO/\\]'

cross_lface = r'(\(|)'
cross_rface = r'(\)|)'
cross_eyes = r'(\^|\*)'
cross_mouth= r'(| |_|-|\.|\\o/)'
cross_smile = '%s%s%s%s%s' % (cross_lface, cross_eyes, cross_mouth, cross_eyes, cross_rface)

right_smile = '(%s|%s)%s(%s|%s)' % (NormalEyes, Wink, NoseArea, HappyMouths, Tongue)

Happy_RE =  mycompile('(%s|%s)' % (cross_smile, right_smile))
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

Emoticon = ('(%s|%s)%s(%s|%s|%s|%s)' % (NormalEyes, Wink, NoseArea, Tongue, OtherMouths, SadMouths, HappyMouths))
Emoticon_RE = mycompile(Emoticon)

def emoticon_parser(text):
    text = Happy_RE.sub(' POS_EMOTICON ', text)
    text = Sad_RE.sub(' NEG_EMOTICON ', text)
    return text


# test known emoticons
if __name__=='__main__':
    emoticon_path = '../../../statics/EmoticonLookupTable.txt'
    with open(emoticon_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        count = 0
        for line in reader:
            d = {
                ' POS_EMOTICON ': 'PositiveSentiment',
                ' NEG_EMOTICON ': 'NegativeSentiment'
            }
            if emoticon_parser(line[0]) not in d or d[emoticon_parser(line[0])] != line[1].strip():
                print('[%s, %s, %s]' % (line[0], emoticon_parser(line[0]), line[1]))
                count += 1
        print(count)
