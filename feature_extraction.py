import numpy as np
from general import get_vocab_dict

pos_prefix = "<词性>:"
dep_prefix = "<依存关系>:"
ROOT = "<root>"
NULL = "<null>"
UNK = "<unk>"


class Dataset(object):  # 用于管理数据集
    def __init__(self, model_config, train_data, dev_data, test_data, feature_extractor):
        self.model_config = model_config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.feature_extractor = feature_extractor

        # Vocab
        self.word2idx = None
        self.idx2word = None
        self.pos2idx = None
        self.idx2pos = None
        self.dep2idx = None
        self.idx2dep = None

        # Embedding Matrix
        self.word_embedding_matrix = None
        self.pos_embedding_matrix = None
        self.dep_embedding_matrix = None

        # input & outputs
        self.train_inputs, self.train_targets = None, None
        self.valid_inputs, self.valid_targets = None, None
        self.test_inputs, self.test_targets = None, None

    def build_vocab(self):  # 建立对应的映射关系

        all_words = set()
        all_pos = set()
        all_dep = set()

        for sentence in self.train_data:
            all_words.update(set(map(lambda x: x.word, sentence.tokens)))
            all_pos.update(set(map(lambda x: x.pos, sentence.tokens)))
            all_dep.update(set(map(lambda x: x.dep, sentence.tokens)))

        all_words.add(ROOT_TOKEN.word)
        all_words.add(NULL_TOKEN.word)

        all_pos.add(ROOT_TOKEN.pos)
        all_pos.add(NULL_TOKEN.pos)

        all_dep.add(ROOT_TOKEN.dep)
        all_dep.add(NULL_TOKEN.dep)

        word_vocab = list(all_words)
        pos_vocab = list(all_pos)
        dep_vocab = list(all_dep)

        word2idx = get_vocab_dict(word_vocab)
        idx2word = {idx: word for (word, idx) in word2idx.items()}

        pos2idx = get_vocab_dict(pos_vocab)
        idx2pos = {idx: pos for (pos, idx) in pos2idx.items()}

        dep2idx = get_vocab_dict(dep_vocab)
        idx2dep = {idx: dep for (dep, idx) in dep2idx.items()}

        self.word2idx = word2idx
        self.idx2word = idx2word

        self.pos2idx = pos2idx
        self.idx2pos = idx2pos

        self.dep2idx = dep2idx
        self.idx2dep = idx2dep


class ModelConfig(object):  # 对模型的维数的管理
    # Input
    word_features_types = None
    pos_features_types = None
    dep_features_types = None
    num_features_types = None
    embedding_dim = 50

    # hidden_size
    l1_hidden_size = 200
    l2_hidden_size = 15

    # output
    num_classes = 3

    # Vocab
    word_vocab_size = None
    pos_vocab_size = None
    dep_vocab_size = None

    # num_epochs
    n_epochs = 20

    # batch_size
    batch_size = 2048

    # dropout
    keep_prob = 0.5
    reg_val = 1e-8

    # learning_rate
    lr = 0.001

    # load existing vocab
    load_existing_vocab = False

    # summary
    write_summary_after_epochs = 1

    # valid run
    run_valid_after_epochs = 1


class Token(object):  # token类，表示一个词，该类目前用于中文
    def __init__(self, token_id, word, pos, dep, head_id):
        self.word = word  # 词本身
        self.token_id = token_id  # 词的序号
        self.pos = pos_prefix + pos  # 词的词性
        self.dep = dep_prefix + dep  # 词的依存关系
        self.head_id = head_id  # 其依赖的中心词的序号
        self.predicted_head_id = None
        self.left_children = list()
        self.right_children = list()

    def is_root_token(self):
        if self.word == ROOT:
            return True
        return False

    def is_null_token(self):
        if self.word == NULL:
            return True
        return False


NULL_TOKEN = Token(-1, NULL, NULL, NULL, -1)
ROOT_TOKEN = Token(-1, ROOT, ROOT, ROOT, -1)
UNK_TOKEN = Token(-1, UNK, UNK, UNK, -1)


class Sentence(object):  # Sentence类，表示一个句子，该类目前用于中文
    def __init__(self, tokens):
        self.Root = Token(-1, ROOT, ROOT, ROOT, -1)
        self.tokens = tokens
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]
        self.dependencies = []
        self.predicted_dependencies = []

    def reset_to_initial_state(self):
        self.buff = [token for token in self.tokens]
        self.stack = [self.Root]

    def get_child_by_index_and_depth(self, token, index, direction, depth):  # 用来找孩子
        if depth == 0:
            return token

        if direction == "left":
            if len(token.left_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.left_children[index]], index, direction, depth - 1)
            return NULL_TOKEN
        else:
            if len(token.right_children) > index:
                return self.get_child_by_index_and_depth(
                    self.tokens[token.right_children[::-1][index]], index, direction, depth - 1)
            return NULL_TOKEN

    def get_legal_labels(self):  # 返回一个长度为3的列表，表示是否有执行三种操作的可能
        labels = ([1] if len(self.stack) > 2 else [0])
        labels += ([1] if len(self.stack) >= 2 else [0])
        labels += [1] if len(self.buff) > 0 else [0]
        return labels

    def get_transition_from_current_state(self):  # 判断该执行什么操作
        if len(self.stack) < 2:
            return 2  # shift

        stack_token_0 = self.stack[-1]
        stack_token_1 = self.stack[-2]
        if stack_token_1.token_id >= 0 and stack_token_1.head_id == stack_token_0.token_id:
            # left arc
            return 0
        elif -1 <= stack_token_1.token_id == stack_token_0.head_id \
                and stack_token_0.token_id not in map(lambda x: x.head_id, self.buff):
            return 1  # right arc
        else:
            return 2 if len(self.buff) != 0 else None

    def update_child_dependencies(self, curr_transition):
        if curr_transition == 0:
            head = self.stack[-1]
            dependent = self.stack[-2]
        elif curr_transition == 1:
            head = self.stack[-2]
            dependent = self.stack[-1]

        if head.token_id > dependent.token_id:
            head.left_children.append(dependent.token_id)
            head.left_children.sort()
        else:
            head.right_children.append(dependent.token_id)
            head.right_children.sort()

    def update_state_by_transition(self, transition, gold=True):  # 更新当前的配置，完成操作
        if transition is not None:
            if transition == 2:  # shift
                self.stack.append(self.buff[0])
                self.buff = self.buff[1:] if len(self.buff) > 1 else []
            elif transition == 0:  # left arc
                self.dependencies.append(
                    (self.stack[-1], self.stack[-2])) if gold else self.predicted_dependencies.append(
                    (self.stack[-1], self.stack[-2]))
                self.stack = self.stack[:-2] + self.stack[-1:]
            elif transition == 1:  # right arc
                self.dependencies.append(
                    (self.stack[-2], self.stack[-1])) if gold else self.predicted_dependencies.append(
                    (self.stack[-2], self.stack[-1]))
                self.stack = self.stack[:-1]


class FeatureExtractor(object):
    def __init__(self, model_config):
        self.model_config = model_config

    def extract_from_stack_and_buffer(self, sentence, num_words=3):
        tokens = []

        tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.stack))])
        tokens.extend(sentence.stack[-num_words:])

        tokens.extend(sentence.buff[:num_words])
        tokens.extend([NULL_TOKEN for _ in range(num_words - len(sentence.buff))])
        return tokens  # 6 features

    def extract_children_from_stack(self, sentence, num_stack_words=2):
        children_tokens = []

        for i in range(num_stack_words):
            if len(sentence.stack) > i:
                lc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left", 1)
                rc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right", 1)

                lc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "left",
                                                            1) if lc0 != NULL_TOKEN else NULL_TOKEN
                rc1 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 1, "right",
                                                            1) if rc0 != NULL_TOKEN else NULL_TOKEN

                llc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "left",
                                                             2) if lc0 != NULL_TOKEN else NULL_TOKEN
                rrc0 = sentence.get_child_by_index_and_depth(sentence.stack[-i - 1], 0, "right",
                                                             2) if rc0 != NULL_TOKEN else NULL_TOKEN

                children_tokens.extend([lc0, rc0, lc1, rc1, llc0, rrc0])
            else:
                [children_tokens.append(NULL_TOKEN) for _ in range(6)]

        return children_tokens  # 12 features

    def extract_for_current_state(self, sentence, word2idx, pos2idx, dep2idx):  # 对当前配置进行特征提取
        direct_tokens = self.extract_from_stack_and_buffer(sentence, num_words=3)  # 得到s0、s1、s2、b0、b1、b2
        children_tokens = self.extract_children_from_stack(sentence, num_stack_words=2)  # 得到s0、s1的12个孩子

        word_features = []
        pos_features = []
        dep_features = []

        # 18个词
        word_features.extend(map(lambda x: x.word, direct_tokens))
        word_features.extend(map(lambda x: x.word, children_tokens))

        # 18个词性
        pos_features.extend(map(lambda x: x.pos, direct_tokens))
        pos_features.extend(map(lambda x: x.pos, children_tokens))

        # 12个依存关系
        dep_features.extend(map(lambda x: x.dep, children_tokens))

        word_input_ids = [word2idx[word] if word in word2idx else word2idx[UNK_TOKEN.word] for word in word_features]
        pos_input_ids = [pos2idx[pos] if pos in pos2idx else pos2idx[UNK_TOKEN.pos] for pos in pos_features]
        dep_input_ids = [dep2idx[dep] if dep in dep2idx else dep2idx[UNK_TOKEN.dep] for dep in dep_features]

        return [word_input_ids, pos_input_ids, dep_input_ids]  # 48个特征

    def create_instances_for_data(self, data, word2idx, pos2idx, dep2idx):  # 生成用于训练模型的训练数据
        lables = []
        word_inputs = []
        pos_inputs = []
        dep_inputs = []
        for i, sentence in enumerate(data):
            num_words = len(sentence.tokens)

            for _ in range(num_words * 2):  # 长度为n的句子需要进行2n次操作
                word_input, pos_input, dep_input = self.extract_for_current_state(sentence, word2idx, pos2idx, dep2idx)
                legal_labels = sentence.get_legal_labels()
                curr_transition = sentence.get_transition_from_current_state()
                if curr_transition is None:
                    break
                assert legal_labels[curr_transition] == 1

                # Update left/right children
                if curr_transition != 2:
                    sentence.update_child_dependencies(curr_transition)

                sentence.update_state_by_transition(curr_transition)
                lables.append(curr_transition)
                word_inputs.append(word_input)
                pos_inputs.append(pos_input)
                dep_inputs.append(dep_input)

            else:
                sentence.reset_to_initial_state()

            # reset stack and buffer to default state
            sentence.reset_to_initial_state()

        targets = np.zeros((len(lables), self.model_config.num_classes), dtype=np.int32)
        targets[np.arange(len(targets)), lables] = 1

        return [word_inputs, pos_inputs, dep_inputs], targets


class DataReader(object):
    def __init__(self):
        print("……")

    def read_conll(self, token_lines):  # 输入为单个句子的行列表，返回一个sentence
        tokens = []
        for each in token_lines:
            fields = each.strip().split("\t")
            token_index = int(fields[0]) - 1
            word = fields[1]
            pos = fields[4]
            dep = fields[7]
            head_index = int(fields[6]) - 1
            token = Token(token_index, word, pos, dep, head_index)
            tokens.append(token)
        sentence = Sentence(tokens)
        return sentence

    def read_data(self, data_lines):  # 对整个数据集读取，遇到换行就调用一次read_conll，返回一个包含所有句子的列表
        data_objects = []
        token_lines = []
        for token_conll in data_lines:
            token_conll = token_conll.strip()  # 去掉头尾的空格
            if len(token_conll) > 0:
                token_lines.append(token_conll)
            else:
                data_objects.append(self.read_conll(token_lines))
                token_lines = []
        if len(token_lines) > 0:
            data_objects.append(self.read_conll(token_lines))
        return data_objects

    def load_datasets(load_existing_dump=False):  # 载入数据集
        model_config = ModelConfig()

        data_reader = DataReader()
        train_lines = open('C:/pyproject/goodparser/train.txt', 'r', encoding='utf-8').readlines()
        valid_lines = open('C:/pyproject/goodparser/dev.txt', 'r', encoding='utf-8').readlines()
        test_lines = open('C:/pyproject/goodparser/test.txt', 'r', encoding='utf-8').readlines()

        # Load data
        train_data = data_reader.read_data(train_lines)
        print("训练集已读取")
        valid_data = data_reader.read_data(valid_lines)
        print("验证集已读取")
        test_data = data_reader.read_data(test_lines)
        print("测试集已读取")
        feature_extractor = FeatureExtractor(model_config)
        dataset = Dataset(model_config, train_data, valid_data, test_data, feature_extractor)

        return dataset
