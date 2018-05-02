import tensorflow as tf

hparam = tf.contrib.training.HParams(
    model_dir='./models',
    data_dir= './data',
    train_file = './data/train.txt',
    eval_file = './data/dev.txt',
    predict_file = './data/test.txt',
    word2vec = './word2vec/GoogleNews-vectors-negative300.bin',
    emb_size = 300,
    seed = 100,
    lr = 0.02,
    hidden_size = 300,
    min_eval_frequency = 100,
    eval_batch_size = 100,
    train_steps =1098734,
    batch_size = 50,
    num_eval_examples = 9840,
    train_sum = 549367,
    keep_prob = 0.9,
    word2vec_type = 'google',
    grad_clip = 5.0
    )