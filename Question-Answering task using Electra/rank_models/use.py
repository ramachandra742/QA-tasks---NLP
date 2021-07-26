import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
USE_PATH = '/Users/renato/Documents/deep_learning/TensorFlow/USE/5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    model = hub.load(USE_PATH)
    use = model(sentences)
    session = tf.train.MonitoredTrainingSession()
    # r = session.run(use, {sentences: a})

def encode(x):
    return session.run(use, {sentences: x})


def cosine(A, B):
    return np.dot(A, B.T) / (np.sqrt(np.sum(A * A)) * np.sqrt(np.sum(B * B)))

def get_similarity(query, documents):
    q = encode(query)
    doc = encode(documents)
    res = []
    for i, v in enumerate(doc):
        res.append(tuple([i, cosine(q, v.reshape(1, -1))[0][0]]))
    res.sort(key=lambda x: x[1], reverse=True)

    return res
