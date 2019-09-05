import json
import os
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation
import numpy as np
import random
#np.set_printoptions(threshold=np.inf)

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf/models")

x_train = np.asarray([[30, 30, 160], [30, 25, 150], [20, 30, 180], [60, 40, 90], [50, 45, 300], [40, 0, 250]])
print(x_train.shape)

y_train = np.asarray([0, -.2, .3, -.7, 0., -.1])
#y_train = np.asarray([[5, 5]])
print(y_train.shape)

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-5)
#opt = tf.keras.optimizers.RMSprop(0.001)

model = Sequential([
    Dense(64, activation=tf.nn.relu, input_shape=(x_train.shape[1:])),
    Dense(32, activation=tf.nn.relu),
    Dense(16, activation=tf.nn.relu),
    Dense(8, activation=tf.nn.relu),
    Dense(1),
  ])

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_squared_error'])

model.fit(x_train, y_train, epochs=100)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    session.run(tf.global_variables_initializer())
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

print(model.predict(np.asarray([[30, 30, 160]])))
#model.save("/Git/dynamic-follow-tf/dynamic_follow_model_test")

frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "/Git/dynamic-follow-tf/models/test/keras", "my_model.pb", as_text=False)