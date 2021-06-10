import feature_extraction as fe
import parser_mode as pm

dt = fe.DataReader().load_datasets()
dt.build_vocab()  # 建立映射关系
train = dt.feature_extractor.create_instances_for_data(dt.train_data, dt.word2idx, dt.pos2idx, dt.dep2idx)
p=pm.parser().model(train)
