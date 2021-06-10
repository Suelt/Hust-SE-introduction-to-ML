
import re
import codecs
#import pandas

WORD = 1
MEANING = 0
ERROR = 1
NOERROR = 0

class word:
    def __init__(self, word, pronunciation, property, meanings, examples, org_content, error):
        self.word = word
        self.pronunciation = pronunciation
        self.property = property
        self.meanings = meanings
        self.examples = examples
        self.org_content = org_content
        self.error = error

# score 1 to word, 0 to meaning
def score_line(pattern, line):
    if len(line) > 35:
        return MEANING
    if pattern.match(line) is None:
        return MEANING
    if line != pattern.match(line).group(0):
        return MEANING
    return WORD


# correct misclassified word based on previous 2 and next 2 line
def correct_misclassified_word(line_score):
    for i in range(2, len(line_score)-2):
        current_line_score = line_score[i][1]
        if current_line_score == 0:
            pp_line_score = line_score[i-2][1]
            prev_line_score = line_score[i-1][1]
            next_line_score = line_score[i+1][1]
            nn_line_score = line_score[i+2][1]
            if (pp_line_score == WORD) and (prev_line_score == 0) and (next_line_score == 0) and (nn_line_score) == 1:
                line_score[i][1] = WORD


# remove end /r/n and empty line and whitespace ending
def preprocess(lines):
    result = []
    for line in lines:
        l = line.strip('\n').strip('\r').strip()
        if l != '':
            result.append(l)
    return result


def split_meaning(meaning):
    return


def extract_word_and_meaning(lines):
    errors = []
    words = []
    meanings = []
    prev_score = MEANING
    for i in range(len(line_score)):
        current_score = line_score[i][1]
        if prev_score == current_score:
            errors.append([line_score[i][0], line_score[i][1], i])
        if prev_score == WORD and current_score == MEANING:
            words.append(line_score[i-1][0])
            meanings.append(line_score[i][0])
        prev_score = current_score
    return words, meanings, errors


if __name__ == "__main__":
    txt_filename = 'ox-edict-utf8.txt'
    pattern_for_word = re.compile(r'^[a-zA-Z\(\)\-\'\`_, ]+$')
    pattern_for_multimean = re.compile(r'(\. 2)|(\(a\))')
    pattern_for_pronunciation = re.compile(r'^(\/[^\/]+\/)?')
    pattern_for_property = re.compile(r'\[.{1,8}\]')
    # read file and preprocess
    lines = []
    with codecs.open(txt_filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
    lines = preprocess(lines)

    # classify each line to word or meaning
    line_score = []
    for line in lines:
        line_score.append([line, score_line(pattern_for_word, line)])
    correct_misclassified_word(line_score)

    # split word and meaning based on classified result
    words, meanings, errors = extract_word_and_meaning(lines)

    multi = []
    for meaning in meanings:
        m = pattern_for_multimean.findall(meaning)
        if m != []:
            multi.append([meaning, 1])
        else:
            multi.append([meaning, 0])
    test_pronunciation = []
    errors_pronunciation = []
    for meaning in meanings:
        pronunciation = pattern_for_pronunciation.search(meaning)
        if pronunciation != None:
            test_pronunciation.append(pronunciation.group(0))
        else:
            errors_pronunciation.append(meaning)
            test_pronunciation.append('')

    test_property = []
    errors_property = []
    for meaning in meanings:
         property = pattern_for_property.findall(meaning)
         if property != []:
             test_property.append(property)
         else:
             errors_property.append(meaning)
             test_property.append('')