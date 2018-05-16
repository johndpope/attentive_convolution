from hparam import hparam
from util import SubDataSet
import tensorflow as tf
import numpy as np
from model import attention_cnn
import json
import os

def get_experiment_fn(dataset, word_emb_matrix, hparam):
  def _experiment_fn(run_config, hparam):

    '''
    if hparam.num_eval_examples % hparam.eval_batch_size != 0:
      raise ValueError(
          'validation set size must be multiple of eval_batch_size')
    '''
    train_steps = hparam.train_sum // hparam.batch_size * 200
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

def set_env(task_type, index):
  cluster = {
              "ps": ["192.168.11.80:2222"],
              "master": ["192.168.11.80:2223"],
              "worker": ["192.168.11.38:2224"]
           }
  TF_CONFIG =  {
      "cluster": cluster,
      "task": {"type": task_type, "index": index},
      'environment': 'cloud'
    }
  print TF_CONFIG
  os.environ["TF_CONFIG"] = json.dumps(TF_CONFIG)

tf.app.flags.DEFINE_string("task", "worker", "task type'")
tf.app.flags.DEFINE_integer("index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("env", 0, "cloud(0) or local(0)")

FLAGS = tf.app.flags.FLAGS

def start_ps(run_config):
  server = tf.train.Server(run_config.cluster_spec,
                         job_name=run_config.task_type,
                         task_index=run_config.task_id)
  print 'ps server started'
  server.join()

def load_emb(dataset):
  if not hparam.voc:
    dataset.generate_vocab()
  else:
    dataset.load_voc_pickle()

  if not hparam.emb_file:
    word_emb = dataset.save_emb_pickle()
  else:
    word_emb = dataset.load_emb_pickle()
  
  return word_emb

def main(_):
  if FLAGS.env == 1:
    set_env(FLAGS.task, FLAGS.index)

  sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  run_config = tf.contrib.learn.RunConfig(session_config=sess_config, model_dir=hparam.model_dir, save_checkpoints_steps=hparam.min_eval_frequency)

  if FLAGS.task == 'ps' and FLAGS.env == 1:
    start_ps(run_config)

  dataset = SubDataSet(hparam)
  word_emb = load_emb(dataset)
  tf.contrib.learn.learn_runner.run(
      get_experiment_fn(dataset, word_emb, hparam),
      run_config=run_config,
      hparams=hparam)


if __name__ == "__main__":
        tf.app.run(main)
