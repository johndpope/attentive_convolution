from hparam import hparam
from util import SubDataSet
import tensorflow as tf
import numpy as np
from model import attention_cnn

def get_experiment_fn(dataset, word_emb_matrix, hparam):
  def _experiment_fn(run_config, hparam):

    '''
    if hparam.num_eval_examples % hparam.eval_batch_size != 0:
      raise ValueError(
          'validation set size must be multiple of eval_batch_size')
    '''
    train_steps = hparam.train_steps
    eval_steps = hparam.num_eval_examples // hparam.eval_batch_size

    attention_cnn_obj = attention_cnn(dataset, hparam)
    classifier = tf.estimator.Estimator(
        model_fn=attention_cnn_obj.get_mode_fn(word_emb_matrix),
        config=run_config,
        params=hparam)

    # Create experiment with the created estimator and return it.
    return tf.contrib.learn.Experiment(
        classifier,
        train_input_fn=dataset.get_train_input, #this func called in the estimator, equal to call input_fn(...)
        eval_input_fn=dataset.get_eval_input,
        train_steps=train_steps,
        eval_steps=eval_steps)

  return _experiment_fn

def main(_):
  dataset = SubDataSet(hparam)
  dataset.generate_vocab()
  rng = np.random.RandomState(100)
  rand_values = rng.normal(0.0, 0.01, (dataset.vocab_len, hparam.emb_size))
  id2word = {y:x for x,y in dataset.vocab_map.iteritems()}
  print 'load word2ver'
  if hparam.word2vec_type == 'google':
    word2vec = dataset.load_word_voctor_from_google_news()
    print 'init matrix'
    rand_values = dataset.init_word2vec_with_google_embeding(rand_values, id2word, word2vec)
  else :
    word2vec = dataset.load_word_voctor_from_glove()
    print 'init matrix'
    rand_values = dataset.init_wordd2vec_with_glove_embding(rand_values, id2word, word2vec)

  run_config = tf.contrib.learn.RunConfig(model_dir=hparam.model_dir, save_checkpoints_steps=hparam.min_eval_frequency)
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(dataset, rand_values, hparam),
      run_config=run_config,
      hparams=hparam)


if __name__ == "__main__":
        tf.app.run(main)
