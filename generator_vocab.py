from hparam import hparam
from util import SubDataSet
import tensorflow as tf

def main(_):
    dataset = SubDataSet(hparam)
    dataset.generate_vocab()
    print 'save dict sucess'
    dataset.save_emb_pickle()
    print 'save word_emb sucess'

if __name__ == "__main__":
        tf.app.run(main)