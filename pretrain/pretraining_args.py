# -----------ARGS---------------------
pretrain_train_path = "./sentence.txt"
pretrain_dev_path = "./sentence.txt"
pretrain_data_dir = "./pretrainingData"
chunk_num = 5  # 一个epoch 划分的总chunk数
load_chunks = 1 # 每次训练加载的chunk数
max_seq_length = 512
do_train = True
do_lower_case = True
train_batch_size = 32
eval_batch_size = 32
learning_rate = 1e-4
num_train_epochs = 10
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 41
gradient_accumulation_steps = 1
fp16 = False
loss_scale = 0.
bert_config_json = "./config.json"
vocab_file = "./vocab.txt"
output_dir = "./root/data/test"
masked_lm_prob = 0.15
max_predictions_per_seq = 76
