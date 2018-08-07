import os
import datetime

import h5py
import cv2
import numpy as np
import tensorflow as tf  # Built on version 1.9.

start_dt = str(datetime.datetime.now())[:19].replace(" ", "_")

# Command-line flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('download_data',
                     False,
                     'If true, downloads either dataset "dataset".')
flags.DEFINE_string('dataset',
                    None,
                    'Name of dataset.')
flags.DEFINE_integer('num_epochs',
                     10,
                     'Number of epochs to train for.')
flags.DEFINE_float('learning_rate',
                   0.0001,
                   'Initial learning rate.')
flags.DEFINE_string('activation',
                    'scaled',
                    ('Type of activation function to use: "rectified", '
                     '"scaled", "exp", "leaky", "cap6", "softsign", '
                     '"softplus".'))
flags.DEFINE_string('dropout_loc',
                    'before',
                    'Whether to apply dropout "before" or "after" pooling.')
flags.DEFINE_float('keep_p_conv',
                   0.9,
                   ('The probability that an element of a given tensor is kept'
                    ', for tensors produced by convolution.'))
flags.DEFINE_float('keep_p_fc',
                   0.5,
                   ('The probability that an element of the tensor produced by'
                    'the first fully-connected layer is kept.'))
flags.DEFINE_boolean('do_test',
                     False,
                     ('If true, trains on whole training set and then tests '
                      'resultant model.'))
flags.DEFINE_integer('summary_freq',
                     1,
                     ('Number of training iterations between training '
                      'summary-writing.'))
flags.DEFINE_boolean('save_model',
                     True,
                     ('If true, saves model to "model_checkpoints/{start_dt}"'
                      'at the end of training.'))
flags.DEFINE_string('restore_model',
                    None,
                    ('If not None, will restore to the model pointed to before'
                     'training.'))
flags.DEFINE_boolean('write_mistakes',
                     False,
                     ('If True, images corresponding to misclassified examples'
                      ' are written to disk.'))



# Manage data.
if FLAGS.download_data:
    os.system('if [ ! -d "data" ]; then mkdir data; fi;')
    os.system(f'if [ ! -d "data/{FLAGS.dataset}" ]; then mkdir data/{FLAGS.dataset}; fi;')
    os.system((f'chmod +x get_dataset_scripts/get_{FLAGS.dataset}.sh &&'
               f'./get_dataset_scripts/get_{FLAGS.dataset}.sh'))

# Set input details.
image_size = 128
num_channels = 3
num_classes = 2
num_splits = 3
X = tf.placeholder('float', [None, image_size, image_size, num_channels])
y = tf.placeholder('float', [None, num_classes])
keep_p_conv = tf.placeholder_with_default(1.0, shape=())
keep_p_fc = tf.placeholder_with_default(1.0, shape=())

# Set key hyperparameters.
num_epochs = FLAGS.num_epochs
learning_rate = FLAGS.learning_rate
batch_size = 256
num_hidden_layers = 6
num_filters_per_layer = [num_channels, 64, 128, 256, 512, 1024, 2048]
fc1_layer_size = 2048
filter_sizes = [None, 5, 3, 3, 3, 3, 1]
pooling_kernel_sizes = [None, 2, 2, 2, 2, 2, 2]

# Create dictionaries for the weight and bias parameters.
weights = dict()
for i in range(1, (num_hidden_layers + 1)):
    weights[f'c{i}'] = tf.get_variable(f'wc{i}',
                                       shape=(filter_sizes[i],
                                              filter_sizes[i],
                                              num_filters_per_layer[i - 1],
                                              num_filters_per_layer[i]),
                                       initializer=tf.contrib.layers.xavier_initializer())
# Assumes stride when pooling is 1, and padding when convoluting is 'SAME'.
final_size = image_size
for i in range(1, len(pooling_kernel_sizes)):
    final_size /= pooling_kernel_sizes[i]
weights['fc1'] = tf.get_variable('wfc1',
                                 shape=(final_size * final_size *
                                        num_filters_per_layer[num_hidden_layers],
                                        fc1_layer_size),
                                 initializer=tf.contrib.layers.xavier_initializer())
weights['fc2'] = tf.get_variable('wfc2',
                                 shape=(fc1_layer_size, num_classes),
                                 initializer=tf.contrib.layers.xavier_initializer())

biases = dict()
for i in range(1, (num_hidden_layers + 1)):
    biases[f'c{i}'] = tf.get_variable(f'bc{i}',
                                     shape=(num_filters_per_layer[i]),
                                     initializer=tf.contrib.layers.xavier_initializer())
biases['fc1'] = tf.get_variable('bfc1',
                                shape=(fc1_layer_size),
                                initializer=tf.contrib.layers.xavier_initializer()),
biases['fc2'] = tf.get_variable('bfc2',
                                shape=(num_classes),
                                initializer=tf.contrib.layers.xavier_initializer())


def activate(X):
    """Applies the activation function of choicce to the feature array."""
    if FLAGS.activation == 'rectified':
        return tf.nn.relu(X)
    elif FLAGS.activation == 'scaled':
        return tf.nn.selu(X)
    elif FLAGS.activation == 'exp':
        return tf.nn.elu(X)
    elif FLAGS.activation == 'leaky':
        return tf.nn.leaky_relu(X)
    elif FLAGS.activation == 'cap6':
        return tf.nn.relu6(X)
    elif FLAGS.activation == 'softsign':
        return tf.nn.softsign(X)
    elif FLAGS.activation == 'softplus':
        return tf.nn.softplus(X)
    else:
        raise ValueError(('Activation function choice not one of "rectified", '
                          '"scaled", "exp", "leaky", "cap6", "softsign", '
                          '"softplus".'))


def convolute(X, W, b, stride=1):
    """tf.nn.convolute() wrapper, with bias and activation."""
    # The first and last element of 'strides' is example and
    # channel stride respectively.
    X = tf.nn.conv2d(X, W,
                     strides=[1, stride, stride, 1],
                     padding='SAME')
    X = activate(tf.nn.bias_add(X, b))
    if FLAGS.dropout_loc == 'before':
        X = tf.nn.dropout(X, keep_p_conv)
    return X


def apply_max_pooling(X, kernel_size=2):
    """Applies max pooling to feature array."""
    # Stride of the kernel is always >= its size to prevent
    # overlap of pooling region.
    X = tf.nn.max_pool(X,
                       ksize=[1, kernel_size, kernel_size, 1],
                       strides=[1, kernel_size, kernel_size, 1],
                       padding='SAME')
    if FLAGS.dropout_loc == 'after':
        X = tf.nn.dropout(X, keep_p_conv)
    return X


def get_fc1_layer(conv_layer):
    """Applies the weights and biases to the input feature array before
    dropout."""
    flat_conv_layer = tf.reshape(conv_layer,
                                 [-1,
                                  weights['fc1'].get_shape().as_list()[0]])
    fc_layer = activate(tf.add(tf.matmul(flat_conv_layer,
                                         weights['fc1']),
                               biases['fc1']))
    return tf.nn.dropout(fc_layer, keep_p_fc)


def get_predicted_labels(fc1_layer):
    """Carries out softmax regression and then dropout to the given feature
    array."""
    return tf.add(tf.matmul(fc1_layer,
                            weights['fc2']),
                  biases['fc2'])


def apply_cnn(X, weights, biases):
    """Runs the feature array through the complete CNN, ultimately returning
    the predicted labels."""
    conv_layer = X
    for i in range(1, (num_hidden_layers + 1)):
        conv_layer = convolute(conv_layer,
                               weights[f'c{i}'],
                               biases[f'c{i}'])
        conv_layer = apply_max_pooling(conv_layer,
                                       kernel_size=pooling_kernel_sizes[i])
    fc1_layer = get_fc1_layer(conv_layer)
    return get_predicted_labels(fc1_layer)


predicted_labels = apply_cnn(X, weights, biases)
find_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_labels,
                                                                      labels=y))
tf.summary.scalar('Loss', find_loss)
optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(find_loss)

is_correct_prediction = tf.equal(tf.argmax(predicted_labels, 1),
                                 tf.argmax(y, 1))
find_accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
tf.summary.scalar('Accuracy', find_accuracy)


def get_true_vs_predicted_labels(true, predicted):
    return np.concatenate((np.argmax(true, axis=1).reshape(-1, 1),
                           np.argmax(predicted, axis=1).reshape(-1, 1)),
                          axis=1)


def write_mistake(mistake_image, true_vs_predicted, curr_split_num, mistake_num):
    curr_split_num = curr_split_num if curr_split_num != 0 else 't'
    base_path = f'mistake_images/{start_dt}_{curr_split_num}'
    os.system('if [ ! -d "mistake_images" ]; then mkdir mistake_images; fi;')
    os.system(f'if [ ! -d "{base_path}" ]; then mkdir {base_path}; fi;')
    cv2.imwrite((f'{base_path}/{mistake_num}_t-{int(true_vs_predicted[0])}_'
                 f'p-{int(true_vs_predicted[1])}.jpg'),
                mistake_image)


def get_confusion_matrix(true_vs_predicted_labels):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(true_vs_predicted_labels)):
        confusion_matrix[int(true_vs_predicted_labels[i, 0]),
                         int(true_vs_predicted_labels[i, 1])] += 1
    return confusion_matrix


def get_sensitivity(confusion_matrix):
    tp = confusion_matrix[1, 1]
    fn = confusion_matrix[1, 0]
    return tp / (tp + fn)


def get_specificity(confusion_matrix):
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    return tn / (tn + fp)


merge_summary_nodes = tf.summary.merge_all()
save_model = tf.train.Saver()

base_path = f'data/{FLAGS.dataset}'
if not FLAGS.do_test:
    validation_accuracies = list()
    for curr_split_num in range(1, (num_splits + 1)):
        train_validation_set = h5py.File((f'{base_path}/train_validation_set_'
                                          f'{curr_split_num}.hdf5'),
                                         'r')
        train_X = train_validation_set['train_X']
        train_y = train_validation_set['train_y']
        validation_X = train_validation_set['validation_X']
        validation_y = train_validation_set['validation_y']
        train_mean_sd = train_validation_set['train_mean_sd']
        with tf.Session() as sess:
            if FLAGS.restore_model is None:
                sess.run(tf.global_variables_initializer())
            else:
                meta = tf.train.import_meta_graph(('model_checkpoints/'
                                                  f'{FLAGS.restore_model}.meta'))
                meta.restore(sess, ('model_checkpoints/'
                                    f'{FLAGS.restore_model}'))
            print()
            print(f'Training and validation on split {curr_split_num}.')
            print('-' * 62)
            summary_writer = tf.summary.FileWriter((f'./tensorboard_log/'
                                                    f'validation_mode/{start_dt}'
                                                    f'_split_{curr_split_num}'),
                                       sess.graph)
            summary_counter = 0
            for epoch in range(1, (num_epochs + 1)):
                for batch in range(len(train_X) // batch_size):
                    batch_X = train_X[batch * batch_size:min((batch + 1) * batch_size,
                                                            len(train_X))]
                    batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size,
                                                             len(train_y))]
                    # Backpropagate, then calculate batch loss and accuracy.
                    summary_counter += 1
                    if summary_counter % FLAGS.summary_freq == 0:
                        summary, _, batch_loss, batch_accuracy= sess.run([merge_summary_nodes,
                                                                           optimize,
                                                                           find_loss,
                                                                           find_accuracy],
                                                                          feed_dict={X: batch_X,
                                                                                     y: batch_y,
                                                                                     keep_p_conv: FLAGS.keep_p_conv,
                                                                                     keep_p_fc: FLAGS.keep_p_fc})
                        summary_writer.add_summary(summary, summary_counter)
                    else:
                        _, batch_loss, batch_accuracy = sess.run([optimize,
                                                                  find_loss,
                                                                  find_accuracy],
                                                                 feed_dict={X: batch_X,
                                                                            y: batch_y,
                                                                            keep_p_conv: FLAGS.keep_p_conv,
                                                                            keep_p_fc: FLAGS.keep_p_fc})
                print((f'Epoch {epoch} | '
                       f'training loss: {batch_loss: .5f}, '
                       f'training accuracy: {batch_accuracy: .5f}.'))
            print('-' * 62)
            if FLAGS.save_model:
                save_path = save_model.save(sess,
                                            (f'model_checkpoints/{start_dt}'
                                             f'_{curr_split_num}'))

            validation_loss = list()
            validation_accuracy = list()
            true_vs_predicted_labels = np.zeros((len(validation_X), 2))
            if FLAGS.write_mistakes:
                mistake_num = 0
            # Feeding the whole validation set (about 15000 examples) takes
            # more memory than we have (~11 GB).
            # Therefore, find 'validation_loss' and 'validation_accuracy' by
            # taking batches of the test set.
            for batch in range(len(validation_X)):
                batch_X = validation_X[batch: min(batch + 1,
                                                  len(validation_X))]
                batch_y = validation_y[batch: min(batch + 1,
                                                  len(validation_y))]
                # Calculate batch loss and accuracy.
                batch_loss, batch_accuracy, predicted = sess.run([find_loss,
                                                                  find_accuracy,
                                                                  predicted_labels],
                                                                 feed_dict={X: batch_X,
                                                                            y: batch_y})
                validation_loss.append(batch_loss)
                validation_accuracy.append(batch_accuracy)
                true_vs_predicted_labels[batch] = get_true_vs_predicted_labels(batch_y,
                                                                               predicted)
                if (true_vs_predicted_labels[batch][0] != true_vs_predicted_labels[batch][1]
                        and FLAGS.write_mistakes):
                    mistake_num += 1
                    write_mistake(batch_X[0] * train_mean_sd[1] + train_mean_sd[0],
                                  true_vs_predicted_labels[batch],
                                  curr_split_num,
                                  mistake_num)
            overall_loss = np.mean(validation_loss)
            overall_accuracy = np.mean(validation_accuracy)
            print((f'Validation loss: {overall_loss: .5f}, '
                   f'validation accuracy: {overall_accuracy: .5f}.'))
            confusion_matrix = get_confusion_matrix(true_vs_predicted_labels)
            print("Confusion matrix (true vs predicted):")
            print(confusion_matrix)
            if (num_classes == 2):
                print(f'Specificity: {get_specificity(confusion_matrix): .5f}')
                print(f'Sensitivity: {get_sensitivity(confusion_matrix): .5f}')
            print('-' * 62)
            print()
            validation_accuracies.append(overall_accuracy)
            summary_writer.close()
    print('-' * 62)
    print(f'Validation accuracy (K = {num_splits}):')
    print(f'Mean: {np.mean(validation_accuracies): .5f}')
    print(f'Median: {np.median(validation_accuracies): .5f}')
    print(f'Standard deviation: {np.std(validation_accuracies): .5f}')
    print('-' * 62)
else:
    test_set = h5py.File(f'{base_path}/test_set.hdf5', 'r')
    test_X, test_y = test_set['test_X'], test_set['test_y']
    train_validation_set = h5py.File(f'{base_path}/train_validation_set_1.hdf5',
                                     'r')
    train_X = np.concatenate((train_validation_set['train_X'],
                              train_validation_set['validation_X']))
    train_y = np.concatenate((train_validation_set['train_y'],
                              train_validation_set['validation_y']))
    train_mean_sd = train_validation_set['train_mean_sd']
    with tf.Session() as sess:
        if FLAGS.restore_model is None:
            sess.run(tf.global_variables_initializer())
        else:
            meta = tf.train.import_meta_graph(('model_checkpoints/'
                                               f'{FLAGS.restore_model}.meta'))
            meta.restore(sess, ('model_checkpoints/'
                                f'{FLAGS.restore_model}'))
        summary_writer = tf.summary.FileWriter((f'./tensorboard_log/'
                                                f'test_mode/{start_dt}'),
                                               sess.graph)
        print()
        print('Training and testing on complete dataset.')
        print('-' * 62)
        summary_counter = 0
        for epoch in range(1, (num_epochs + 1)):
            for batch in range(len(train_X) // batch_size):
                batch_X = train_X[batch * batch_size:min((batch + 1) * batch_size,
                                                         len(train_X))]
                batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size,
                                                         len(train_y))]
                # Backpropagate, then calculate batch loss and accuracy.
                summary_counter += 1
                if summary_counter % FLAGS.summary_freq == 0:
                    summary, _, batch_loss, batch_accuracy = sess.run([merge_summary_nodes,
                                                                       optimize,
                                                                       find_loss,
                                                                       find_accuracy],
                                                                      feed_dict={X: batch_X,
                                                                                 y: batch_y,
                                                                                 keep_p_conv: FLAGS.keep_p_conv,
                                                                                 keep_p_fc: FLAGS.keep_p_fc})
                    summary_writer.add_summary(summary, summary_counter)
                else:
                    _, batch_loss, batch_accuracy = sess.run([optimize,
                                                              find_loss,
                                                              find_accuracy],
                                                             feed_dict={X: batch_X,
                                                                        y: batch_y,
                                                                        keep_p_conv: FLAGS.keep_p_conv,
                                                                        keep_p_fc: FLAGS.keep_p_fc})
            print((f'Epoch {epoch} | '
                   f'training loss: {batch_loss: .5f}, '
                   f'training accuracy: {batch_accuracy: .5f}.'))
        print('-' * 62)
        if FLAGS.save_model:
                save_path = save_model.save(sess,
                                            f'model_checkpoints/{start_dt}_t')
        test_loss = list()
        test_accuracy = list()
        true_vs_predicted_labels = np.zeros((len(train_X), 2))
        if FLAGS.write_mistakes:
            mistake_num = 0
        # Feeding the whole test set (about 5000 examples) takes more memory
        # than we have (~11 GB).
        # Therefore, find 'test_loss' and 'test_accuracy' by taking batches of
        # the test set.
        for batch in range(len(test_X)):
            batch_X = test_X[batch: min(batch + 1,
                                        len(test_X))]
            batch_y = test_y[batch: min(batch + 1,
                                        len(test_y))]
            # Calculate batch loss and accuracy.
            batch_loss, batch_accuracy, predicted = sess.run([find_loss,
                                                              find_accuracy,
                                                              predicted_labels],
                                                             feed_dict={X: batch_X,
                                                                        y: batch_y})
            test_loss.append(batch_loss)
            test_accuracy.append(batch_accuracy)
            true_vs_predicted_labels[batch] = get_true_vs_predicted_labels(batch_y,
                                                                           predicted)
            if (true_vs_predicted_labels[batch][0] != true_vs_predicted_labels[batch][1]
                    and FLAGS.write_mistakes):
                mistake_num += 1
                write_mistake(batch_X[0] * train_mean_sd[1] + train_mean_sd[0],
                              true_vs_predicted_labels[batch],
                              0,
                              mistake_num)
        overall_loss = np.mean(test_loss)
        overall_accuracy = np.mean(test_accuracy)
        print((f'Test loss: {overall_loss: .5f}, '
               f'test accuracy: {overall_accuracy: .5f}.'))
        confusion_matrix = get_confusion_matrix(true_vs_predicted_labels)
        print("Confusion matrix (true vs predicted):")
        print(confusion_matrix)
        if (num_classes == 2):
            print(f'Specificity: {get_specificity(confusion_matrix): .5f}')
            print(f'Sensitivity: {get_sensitivity(confusion_matrix): .5f}')
        print('-' * 62)
        summary_writer.close()
print()

