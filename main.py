
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle as plk
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
print(tf.__version__)
# Jie You 
# Deep_Learning Final_Project
# Action recognition with Deep Learning

# parameters
total_size = 7272
train_size = 6222
valid_size = 1050
learning_rate = 0.001
training_iters = 1600
batch_size = 150
display_step = 30
keep_prob = 0.9
training = 1
testing = -1
dropout = 1
num_units = 110
seq_length = 30
num_class = 11


# In[2]:


def preProcess():
    print('starting preprocessing')
    data_file = open('youtube_action_train_data_part1.pkl', 'rb')
    train_x_1, train_y_1 = plk.load(data_file)
    data_file.close()
    
    data_file = open('youtube_action_train_data_part2.pkl', 'rb')
    train_x_2, train_y_2 = plk.load(data_file)
    data_file.close()

    train_x = np.concatenate((train_x_1, train_x_2), axis = 0)
    train_y = np.concatenate((train_y_1, train_y_2), axis = 0)
    
    # converting float32
    train_x = np.float32(train_x)
    train_y = np.int32(train_y)
    
    idx = np.arange(total_size)
    np.random.shuffle(idx)
    
    train_idx = idx[np.arange(train_size)]
    eval_idx = idx[train_size + np.arange(valid_size)]
    
    train_data = train_x[train_idx]
    eval_data = train_x[eval_idx]
    print('normalzing')
    # scaling and normalize
    train_data -= np.mean(train_data, axis = (2, 3, 4), keepdims = True)
    train_data /= np.std(train_data, axis = (2, 3, 4), keepdims = True)
    
    eval_data -= np.mean(eval_data, axis = (2, 3, 4), keepdims = True)
    eval_data /= np.std(eval_data, axis = (2, 3, 4), keepdims = True)
    
    y = train_y[train_idx];
    y_test = train_y[eval_idx];
    
    train_labels = np.zeros((train_size, num_class))
    train_labels[np.arange(train_size), y] = 1
    
    eval_labels = np.zeros((valid_size, num_class))
    eval_labels[np.arange(valid_size), y_test] = 1
    print('preprocessing is done!')
    return train_data, train_labels, eval_data, eval_labels


# In[3]:


def lrelu(x, alpha=0., max_value=None):
    negative_part = tf.nn.relu(-x)
    x = tf.nn.relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x
 
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID') 
    x = tf.nn.bias_add(x, b) 
    #return lrelu(x)
    return tf.nn.relu(x) # 

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID') # 
 
# Create CNN model
def conv_net(training, x, weights, biases, keep_prob): # weights are dictionaries
    # Reshape input picture
    global seq_length
    cnn_out = tf.zeros(shape = [batch_size, 0, 1024])
    for i in np.arange(seq_length) :
        temp = tf.reshape(x[:, i, :, :, :], [-1, 64, 64, 3])
        # Convolution Layer
        conv1 = conv2d(temp, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)
        conv1=tf.cond(training, lambda: tf.nn.dropout(conv1,keep_prob=0.9 if dropout else 1.0) ,lambda: conv1) 
 
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2) 
        conv2=tf.cond(training, lambda: tf.nn.dropout(conv2,keep_prob=0.8 if dropout else 1.0) ,lambda: conv2) 
    
        # Convolution Layer
        conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
        # Max Pooling (down-sampling)
        conv3 = maxpool2d(conv3, k=2)  
        conv3=tf.cond(training, lambda: tf.nn.dropout(conv3,keep_prob=0.7 if dropout else 1.0) ,lambda: conv3) 
        
        # Fully connected layer
        out = tf.reshape(conv3, [-1, 1, 1024])
        cnn_out = tf.concat([cnn_out, out], axis = 1)
     
    return cnn_out

def plot_loss(loss,train_step,from_second,name_save, plot_name,plot_title):
    if from_second :
        plt.plot(range(0,train_step-1,1),loss[1:])
    else:
        plt.plot(range(0,train_step,1),loss[0:])
    plt.xlabel('Iterative times (t)')
    plt.ylabel(plot_name)
    plt.title(plot_title)
    plt.grid(True)
    plt.savefig(name_save)
    plt.show() 


# In[3]:


# In[4]:


def establish_model(train_data,train_labels,eval_data,eval_labels):
    print('establishing model')
    global keep_prob, training, testing, dropout, learning_rate, training_iters, display_step, batch_size, num_units, seq_length, valid_size, num_class
    x = tf.placeholder(tf.float32, [None, seq_length, 64, 64, 3])
    y = tf.placeholder(tf.float32, [None, num_class])
     # Store layers weight & bias
    weights = {
    # 5x5 conv, 3 input, 32 outputs
      'wc1': tf.get_variable( 'weight1',shape = [7, 7, 3, 32],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)),    
    # 5x5 conv, 32 inputs, 32 outputs
      'wc2': tf.get_variable( 'weight2',shape = [6, 6, 32, 32],
        initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32)), #tf.Variable(tf.random_normal([5, 5, 32, 32])),
    # 4x4 conv, 32 inputs, 64 outputs
      'wc3': tf.get_variable( 'weight3',shape = [5, 5, 32, 64],
        initializer = tf.contrib.layers.xavier_initializer_conv2d())
        }
    biases = {
      'bc1': tf.get_variable( 'bias1',
          shape = [32],
          initializer=tf.constant_initializer(0.0)),# tf.Variable(tf.random_normal([28,28,32])),# could not understant the dim
      'bc2': tf.get_variable(  'bias2',
          shape = [32],
          initializer=tf.constant_initializer(0.0)),#tf.Variable(tf.random_normal([10,10,32])),
      'bc3':tf.get_variable(  'bias3',
          shape = [64],
          initializer=tf.constant_initializer(0.0)) #tf.Variable(tf.random_normal([3,3,64])),
        }
    #cnn_out should be [batch_size, 10, 1024]
    #RNN weight
    w_fc = tf.get_variable("w_fc", shape=[num_units, num_class], initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)) 
    b_fc = tf.get_variable("b_fc", shape=[num_class], initializer=tf.constant_initializer(0.0))
        # Construct CNN model
    cnn_out = conv_net(tf.greater(training, 0), x, weights, biases, keep_prob)
    cnn_out_predict = conv_net(tf.greater(testing, 0), x, weights, biases, keep_prob)
    # RNN cell
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    h_val, _ = tf.nn.dynamic_rnn(lstm_cell, cnn_out, dtype = tf.float32)
    h_val_predict, _ = tf.nn.dynamic_rnn(lstm_cell, cnn_out_predict, dtype = tf.float32)
    
    # final output
    
    # Get output of RNN sequence
    temp2 = tf.reshape(h_val[:, seq_length - 1, :], [-1, num_units])
    output = tf.matmul(temp2, w_fc) + b_fc
    pred = tf.reshape(output, [-1, num_class])
    
    predict_op_train = tf.argmax(tf.nn.softmax(pred), 1)
    
    # Loss function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred, labels = y))
    correct = tf.equal(predict_op_train, tf.argmax(y, 1))
    err = 1 - tf.reduce_mean(tf.cast(correct, tf.float32))
    
    # testing predict
    temp3 = tf.reshape(h_val_predict[:, seq_length - 1, :], [-1, num_units])
    output_predict = tf.matmul(temp3, w_fc) + b_fc
    pred_test = tf.reshape(output_predict, [-1, num_class])
    
    predict_op = tf.argmax(tf.nn.softmax(pred_test), 1)
    correct_test = tf.equal(predict_op, tf.argmax(y, 1))
    err_test = 1 - tf.reduce_mean(tf.cast(correct_test, tf.float32))
    
    # Declare optimizers
    step = 1
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate/np.sqrt(step)).minimize(cost)
    
    # evaluation the results
    #err_test = each_perform(y_rotate, final_output, valid_size)
    
    
    # Evaluate model
    #correct = tf.nn.l2_loss(y - final_output)
    loss_history = []
    train_rate_history = []
    val_rate_history = []
    
    # Initializing the variables
     
    #save
    saver = tf.train.Saver()
    # Launch the graph
    #sess = tf.InteractiveSession()
    sess = tf.Session()
    print('sess starting')
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    
    while step < training_iters :
        ind = np.arange(train_data.shape[0])
        batch_idx = np.random.choice(ind, batch_size, replace=False)
        batch_x = train_data[batch_idx]
        batch_y= train_labels[batch_idx]
        indtest = np.arange(eval_data.shape[0])
        test_idx = np.random.choice(indtest, batch_size, replace=False)
        batch_xtest = eval_data[test_idx]
        batch_ytest = eval_labels[test_idx]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y  })
        loss, train_err = sess.run([cost, err], feed_dict={x: batch_x, y: batch_y })
        test_err = sess.run(err_test, feed_dict={x: batch_xtest, y: batch_ytest })
        loss_history.append(loss)
        train_rate_history.append(train_err)
        val_rate_history.append(test_err)
        if step % display_step == 0:
            # Calculate batch loss and err
            print("Iter " + str(step ) + ", Minibatch Loss= " +                   "{:.6f}".format(loss) + ", testing err= " +                   "{:.5f}".format(test_err)+ ", training err= " +                   "{:.5f}".format(train_err))
        step += 1
    

    indtest = np.arange(eval_data.shape[0])
    test_idx = np.random.choice(indtest, batch_size, replace=False)
    batch_xtest = eval_data[test_idx]
    batch_ytest = eval_labels[test_idx]
    test_result = sess.run(correct_test, feed_dict={x: batch_xtest, y: batch_ytest})
    predict_result = sess.run(predict_op, feed_dict={x: batch_xtest, y: batch_ytest})
    test_y = batch_ytest
    
    for j in np.arange(6) :
        indtest = np.arange(eval_data.shape[0])
        test_idx = np.random.choice(indtest, batch_size, replace=False)
        batch_xtest = eval_data[test_idx]
        batch_ytest = eval_labels[test_idx]
        temp_results = sess.run(correct_test, feed_dict={x: batch_xtest, y: batch_ytest })
        predict_rs = sess.run(predict_op, feed_dict={x: batch_xtest, y: batch_ytest})
        test_result = np.concatenate((test_result, temp_results), axis = 0)
        test_y = np.concatenate((test_y, batch_ytest), axis = 0)
        predict_result = np.concatenate((predict_result, predict_rs), axis = 0);
    
    label_y = np.argmax(test_y, 1)
    confusion = tf.confusion_matrix(labels = label_y, predictions = predict_result, num_classes = num_class)
    
   
    #print("Testing correct:", correct_results) 
    #print("Testing err:", test_error)    
    print("Optimization Finished!")
    # Create the collection.
    tf.get_collection("validation_nodes")
    # Add stuff to the collection.
    tf.add_to_collection("validation_nodes", x)
    tf.add_to_collection("validation_nodes", y)
    tf.add_to_collection("validation_nodes", predict_op) 
    save_path = saver.save(sess, "./my-model/my-model.ckpt")   

    return test_result, test_y, loss_history, train_rate_history, val_rate_history, step, confusion


# In[4]:


def each_perform(correct_results, eval_labels, num_test):
    label_y = np.argmax(eval_labels,1)
    
    zero = np.where(label_y==0)
    correct=[correct_results[i] for i in zero]
    accuracy_zero= np.mean(correct)

    one = np.where(label_y==1)
    correct=[correct_results[i] for i in one]
    accuracy_one= np.mean(correct)

    two = np.where(label_y==2)
    correct=[correct_results[i] for i in two]
    accuracy_two= np.mean(correct)

    three = np.where(label_y==3)
    correct=[correct_results[i] for i in three]
    accuracy_three= np.mean(correct)

    four = np.where(label_y==4)
    correct=[correct_results[i] for i in four]
    accuracy_four= np.mean(correct)

    five = np.where(label_y==5)
    correct=[correct_results[i] for i in five]
    accuracy_five= np.mean(correct)

    six = np.where(label_y==6)
    correct=[correct_results[i] for i in six]
    accuracy_six= np.mean(correct)

    seven = np.where(label_y==7)
    correct=[correct_results[i] for i in seven]
    accuracy_seven= np.mean(correct)

    eight = np.where(label_y==8)
    correct=[correct_results[i] for i in eight]
    accuracy_eight= np.mean(correct)

    nine = np.where(label_y==9)
    correct=[correct_results[i] for i in nine]
    accuracy_nine= np.mean(correct)
    
    ten = np.where(label_y==10)
    correct=[correct_results[i] for i in ten]
    accuracy_ten= np.mean(correct)
    
    print(accuracy_zero)
    print(accuracy_one)
    print(accuracy_two)
    print(accuracy_three)
    print(accuracy_four)
    print(accuracy_five)
    print(accuracy_six)
    print(accuracy_seven)
    print(accuracy_eight)
    print(accuracy_nine)
    print(accuracy_ten)


# In[5]:


def main():
    train_data, train_labels, eval_data, eval_labels = preProcess()
    test_result, test_y, loss_history, train_rate_history, val_rate_history, step, confusion = establish_model(train_data,train_labels,eval_data,eval_labels)
    plot_loss(loss_history, step-1, False, "Loss_value.png", 'Loss','Loss function value with iterations') 
    plot_loss(train_rate_history, step-1, False, "Train_error_value.png", 'Training error','Training error function value with iterations') 
    plot_loss(val_rate_history, step-1, False,"Test_error_value.png",'Testing error','Testing error function with iterations')
    each_perform(test_result, test_y, 7 * batch_size)
    with tf.Session() as sess:  
        print(confusion.eval())  
    
if __name__ == '__main__':
    main()

