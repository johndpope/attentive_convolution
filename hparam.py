import tensorflow as tf

hparam = tf.contrib.training.HParams(
    model_dir='./models',
    data_dir= './data',
    train_file = './data/train.txt',
    eval_file = './data/dev.txt',
    predict_file = './data/test.txt',
    google_word2vec = './word2vec/GoogleNews-vectors-negative300.bin',
    glove_word2vec = './word2vec/glove.6B.100d.txt',
    emb_size = 100,
    seed = 100,
    lr = 0.001,
    hidden_size = 100,
    eval_batch_size = 100,
    batch_size = 32,
    num_eval_examples = 9840,
    min_eval_frequency = 1000,
    train_sum = 549367,
    keep_prob = 0.9,
    word2vec_type = 'glove',
    grad_clip = 5.0,
    l2_c = 3e-6,
    voc='./voc_emb/voc',
    emb_file = './voc_emb/word_embeding',
    opt ='adm',
    buffer_size = 2000
    )