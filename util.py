import tensorflow as tf
import numpy as np
#from tensorflow.contrib.data import Dataset, TextLineDataset
import os
import chardet
from sklearn.feature_extraction.text import CountVectorizer
import string
import gensim

class SubDataSet(object):
    """docstring for Dataset"""
    def __init__(self, hparam):
        self.path = hparam.data_dir
        self.hparam = hparam

    def get_train_input(self):
        dataset = tf.data.TextLineDataset(self.hparam.train_file).map(self.tokenize)
        dataset = dataset.repeat() \
            .padded_batch(
                self.hparam.batch_size,
                padded_shapes=([None], [None], [None], [None], [None]))
        iterator = dataset.make_one_shot_iterator()
        left_ids, left_mask, right_ids, right_mask, labels = iterator.get_next()
        feature ={
                    'orig_input_left': left_ids,
                    'orig_input_left_mask': left_mask,
                    'orig_input_right': right_ids,
                    'orig_input_right_mask' : right_mask,
                    }
        return feature, labels

    def get_eval_input(self):
        dataset = tf.data.TextLineDataset(self.hparam.eval_file).map(self.tokenize)
        dataset = dataset.repeat() \
            .padded_batch(
                self.hparam.eval_batch_size,
                padded_shapes=([None], [None], [None], [None], [None]))
        iterator = dataset.make_one_shot_iterator()
        left_ids, left_mask, right_ids, right_mask, labels = iterator.get_next()
        feature ={
                    'orig_input_left': left_ids,
                    'orig_input_left_mask': left_mask,
                    'orig_input_right': right_ids,
                    'orig_input_right_mask' : right_mask,
                    }
        return feature, labels

    def get_predict_input(self):
        dataset = tf.data.TextLineDataset(self.hparam.predict_file).map(self.tokenize)
        dataset = dataset.repeat() \
            .padded_batch(
                self.hparam.eval_batch_size,
                padded_shapes=([None], [None], [None], [None], [None]))
        iterator = dataset.make_one_shot_iterator()
        left_ids, left_mask, right_ids, right_mask, labels = iterator.get_next()
        feature = {
                    'orig_input_left': left_ids,
                    'orig_input_left_mask': left_mask,
                    'orig_input_right': right_ids,
                    'orig_input_right_mask' : right_mask,
                    }
        return feature, labels

    def translate_lable_to_onehot(self, label_str):
        label = label_str.strip()
        if label == 'neutral':
            return [1.0,0,0]
        elif label == 'contradiction':
            return [0,1.0,0]
        else :
            return [0,0,1.0]

    def tokenize(self, line):
        def _tokenize(line):
            parts=line.strip().split('\t')
            label=self.translate_lable_to_onehot(parts[2])  # keep label be 0 or 1
            sentence_wordlist_l=parts[0].strip().lower().split()
            sentence_wordlist_r=parts[1].strip().lower().split()
            left_ids = []
            right_ids = []

            #src-sentence
            for word in sentence_wordlist_l:
                position = word.find('-')
                if position<0:
                    if word not in string.punctuation:
                        word =  word.translate(None, string.punctuation)
                        wordid = self.vocab_map.get(word, self.vocab_map['_UNK']) 
                        left_ids.append(wordid)
                else:
                    subwords = word.split('-')
                    for subword in subwords:
                        if subword not in string.punctuation:
                            subword =  subword.translate(None, string.punctuation)
                            wordid = self.vocab_map.get(word, self.vocab_map['_UNK']) 
                            left_ids.append(wordid)
            left_ids = np.array(left_ids, dtype=np.int32)
            left_mask = np.ones(left_ids.shape[0], dtype=np.float32)

            #target sentence
            for word in sentence_wordlist_r:
                position = word.find('-')
                if position<0:
                    if word not in string.punctuation:
                        word =  word.translate(None, string.punctuation)
                        wordid = self.vocab_map.get(word, self.vocab_map['_UNK']) 
                        right_ids.append(wordid)
                else:
                    subwords = word.split('-')
                    for subword in subwords:
                        if subword not in string.punctuation:
                            subword =  subword.translate(None, string.punctuation)
                            wordid = self.vocab_map.get(word, self.vocab_map['_UNK']) 
                            right_ids.append(wordid)
            right_ids = np.array(right_ids, dtype=np.int32)
            right_mask = np.ones(right_ids.shape[0], dtype=np.float32)

            label = np.array(label, dtype=np.float32)
            return left_ids, left_mask, right_ids, right_mask, label

        return tuple(tf.py_func(_tokenize,
                                [line],
                                [tf.int32, tf.float32, tf.int32, tf.float32, tf.float32],
                                ))

    def listdir(self, path, list_name): 
      for file in os.listdir(path): 
        file_path = os.path.join(path, file) 
        if os.path.isdir(file_path): 
          self.listdir(file_path, list_name) 
        else:
          list_name.append(file_path) 

    def generate_vocab(self):
        self.vocab_map = self.preprocess_files(max_f = 0.1, min_f = 5)
        self.vocab_len = len(self.vocab_map)

    def preprocess_files(self, max_f=1.0, min_f=1):
        vocab_map = {}
        filelist = []
        self.listdir(self.path, filelist)

        vocab_len_dic = {}
        wholeword = 0
        for file in filelist:
            fi = open(file)
            for line in fi:
                line = line.strip().lower().split()
                for word in line:
                    position = word.find('-')
                    if position<0:
                        if word not in string.punctuation:
                            word =  word.translate(None, string.punctuation)
                        wholeword += 1
                        if vocab_map.get(word) == None:
                            vocab_map[word] = 1
                            vocab_len_dic[word] = 1
                        else:
                            vocab_len_dic[word] += 1
                    else:
                        subwords = word.split('-')
                        for subword in subwords:
                            if subword not in string.punctuation:
                                subword =  subword.translate(None, string.punctuation)
                            wholeword += 1
                            if vocab_map.get(subword) == None:
                                vocab_map[subword] = 1
                                vocab_len_dic[subword] = 1
                            else:
                                vocab_len_dic[subword] += 1
            fi.close()

        words_to_drop = set()
        for key in vocab_len_dic:
            if vocab_len_dic[key] < min_f:
                words_to_drop.add(key)
                continue

            if vocab_len_dic[key] / wholeword > max_f:
                words_to_drop.add(key)

        for key in words_to_drop:
            del vocab_map[key]
        '''
        i = 0
        for word in words_to_drop:
            if i < 20 :
                print word
                i+=1
                continue
            break
        '''
        id = 1
        for key in vocab_map:
            vocab_map[key] = id
            id += 1

        vocab_map['_UNK'] = 0
        print 'vocab len: ', len(vocab_map)
        return vocab_map

    def load_word_voctor_from_google_news(self):
        word2vec = gensim.models.KeyedVectors.load_word2vec_format(self.hparam.word2vec, binary=True) 
        return word2vec

    def load_word_voctor_from_glove(self):
        word2vec = {}
        
        print "==> loading word2vec"
        f = open(self.hparam.word2vec, 'r')

        for line in f:
            l = line.split()
            word2vec[l[0]] = map(float, l[1:])

        print "==> word2vec is loaded"
        

    def init_wordd2vec_with_glove_embding(self, rand_values, ivocab, word2vec):
        fail=0
        sucess = 0
        print 'word2vec len: ', len(word2vec)
        print 'vocab_len: ', len(ivocab)
        for id, word in ivocab.iteritems():
            emb=word2vec.get(word)
            if emb is not None:
                sucess+=1
                rand_values[id]=np.array(emb,dtype=np.float32)
            else:
                if fail < 60:
                    print 'fail: word=', word, 'id=', id
                # print word
                fail+=1
        print '==> use word2vec initialization over...fail ', fail
        print 'sucess:', sucess
        return rand_values

    def init_word2vec_with_google_embeding(self, rand_values, ivocab, word2vec):
        fail=0
        sucess = 0
        print 'vocab_len: ', len(ivocab)
        for id, word in ivocab.iteritems():
            if word in word2vec.vocab:
                emb=word2vec[word]
                sucess+=1
                rand_values[id] = emb#np.array(emb,dtype=np.float32)
            else:
                if fail < 60:
                    print 'fail: word=', word, 'id=', id
                # print word
                fail+=1
        print '==> use word2vec initialization over...fail ', fail
        print 'sucess:', sucess
        return rand_values