import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.nn import static_rnn
from tensorflow.contrib.rnn import BasicLSTMCell
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
tf.reset_default_graph()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

#data = pd.read_csv('creditcard.csv', skiprows=[0], header=None)

#features = data.iloc[:, 1:30]
#labels = data.iloc[:, -1]


#x = np.array([[[.1], [.2]], [[.4], [.5]], [[.3], [.4]], [[.8], [.9]]], dtype=np.float32)
x = np.array([[.1, .2], [.4, .5], [.5, .6], [.6, .7], [.7, .8], [.1, .2], [0, .1], [.45, .55], [.05, .15], [.77, .87], [.23, .33]], dtype=np.float32)
y = np.array([.3, .6, .7, .8, .9, .3, .2, .65, .25, .97, .43], dtype=np.float32)

X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.5, shuffle=True)

#data = pd.read_csv('C:/Users/Shane/creditcard.csv', skiprows=[0], header=None)
#features = data.iloc[:, 1:30]
#labels = data.iloc[:, -1]

#X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)

epochs = 200
n_classes = 1
n_units = 64
n_features = 2
batch_size = 1

xplaceholder= tf.placeholder('float',[None,n_features])
yplaceholder = tf.placeholder('float')

def recurrent_neural_network_model():
    layer ={ 'weights': tf.Variable(tf.random_normal([n_units, n_classes])),'bias': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.split(xplaceholder, n_features, 1)
    print(x)

    lstm_cell = BasicLSTMCell(n_units)
    
    outputs, states = static_rnn(lstm_cell, x, dtype=tf.float32)
   
    output = tf.matmul(outputs[-1], layer['weights']) + layer['bias']

    return output


logit = recurrent_neural_network_model()
logit = tf.reshape(logit, [-1])

#cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit, labels=yplaceholder))
cost = tf.reduce_mean(tf.square(logit - yplaceholder))
optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.Session() as sess:

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    for epoch in range(epochs):
        epoch_loss = 0

        i = 0
        for i in range(int(len(X_train) / batch_size)):

            start = i
            end = i + batch_size

            batch_x = np.array(X_train[start:end])
            #print(batch_x)
            batch_y = np.array(y_train[start:end])
            _, c = sess.run([optimizer, cost], feed_dict={xplaceholder: batch_x, yplaceholder: batch_y})
            epoch_loss += c
            i += batch_size

        print('Epoch', epoch, 'completed out of', epochs, 'loss:', epoch_loss)

    pred = tf.round(tf.nn.sigmoid(logit)).eval({xplaceholder: np.array(X_test), yplaceholder: np.array(y_test)})
    
    for idx, i in enumerate(X_test):
        print('Truth: {}'.format(y_test[idx]))
        print('Prediction: {}'.format(sess.run(logit, feed_dict={xplaceholder: [X_test[idx]]})[0]))
    
    frozen_graph = freeze_session(sess)
    print([n.name for n in frozen_graph.node])

    tf.train.write_graph(frozen_graph, 'C:/Git/dynamic-follow-tf/models/pb_models', 'saved_model.pb', as_text=False)

    '''f1 = f1_score(np.array(y_test), pred, average='macro')
    accuracy=accuracy_score(np.array(y_test), pred)
    recall = recall_score(y_true=np.array(y_test), y_pred= pred)
    precision = precision_score(y_true=np.array(y_test), y_pred=pred)
    print("F1 Score:", f1)
    print("Accuracy Score:",accuracy)
    print("Recall:", recall)
    print("Precision:", precision)'''
