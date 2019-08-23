import tensorflow as tf
import numpy as np

n_nodes_h1 = 100
n_nodes_h2 = 100
n_nodes_h3 = 100
n_nodes_h4 = 100
n_classes = 2

train_data = []
train_labels = []
test_data = []
test_labels = []

f = open('C:\\Users\\DELL-PC\\Desktop\\Deep_Data Set.txt','r')
i=0
for line in f.readlines():
    line = line.split(',')
    i+=1  
    if(i<456):
        if(line[1] == 'M'):
            train_labels.append([float(1),float(0)])
        elif(line[1] == 'B'):
             train_labels.append([float(0),float(1)])
        del line[1]
        del line[0]
        train_data.append([np.float32(j) for j in line])
        
    else:
        if(line[1] == 'M'):
            test_labels.append([float(1),float(0)])
        elif(line[1] == 'B'):
             test_labels.append([float(0),float(1)])
        del line[1]
        del line[0]
        test_data.append([np.float32(j) for j in line])
                

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)


print(train_data.shape,train_labels.shape)
print(test_data.shape,test_labels.shape)

x = tf.placeholder(tf.float32,[None,30])
y = tf.placeholder(tf.float32,[None,2])

hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([30,n_nodes_h1], stddev=0.01, dtype=tf.float32)),'biases':tf.Variable(tf.random_normal([n_nodes_h1], stddev=0.01, dtype=tf.float32))}
hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_h1,n_nodes_h2], stddev=0.01, dtype=tf.float32)),'biases':tf.Variable(tf.random_normal([n_nodes_h2], stddev=0.01, dtype=tf.float32))}
#hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_h2,n_nodes_h3], stddev=0.01, dtype=tf.float32)),'biases':tf.Variable(tf.random_normal([n_nodes_h3], stddev=0.01, dtype=tf.float32))}
#hidden_layer_4 = {'weights':tf.Variable(tf.random_normal([n_nodes_h3,n_nodes_h4], stddev=0.01, dtype=tf.float32)),'biases':tf.Variable(tf.random_normal([n_nodes_h3], stddev=0.01, dtype=tf.float32))}
output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_h2,n_classes], stddev=0.01, dtype=tf.float32)),'biases':tf.Variable(tf.random_normal([n_classes], stddev=0.01, dtype=tf.float32))}

layer1 = tf.add(tf.matmul(x ,hidden_layer_1['weights']),hidden_layer_1['biases'])
layer1 = tf.nn.relu(layer1)

layer2 = tf.add(tf.matmul(layer1,hidden_layer_2['weights']),hidden_layer_2['biases'])
layer2 = tf.nn.relu(layer2)

#layer3 = tf.add(tf.matmul(layer2,hidden_layer_3['weights']),hidden_layer_3['biases'])
#layer3 = tf.nn.tanh(layer3)

#layer4 = tf.add(tf.matmul(layer3,hidden_layer_4['weights']),hidden_layer_4['biases'])
#layer4 = tf.nn.tanh(layer4)

output = tf.add(tf.matmul(layer2,output_layer['weights']), output_layer['biases'])
    

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output,labels = y))
optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(cost)

number_epochs = 50000
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(number_epochs+1):
        sess.run(train,feed_dict={x:train_data, y:train_labels})
            
        if i%10000 == 0:
            #print('Epoch = ',str(i+1), 'Loss = ',str(epoch_loss))
            correct = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float32'))
            print('Epoch No:',str(i),'Train Accuracy:',accuracy.eval({x:train_data,y:train_labels}))
            #print(sess.run(tf.nn.softmax(output), feed_dict= {x:a}))
            #print(sess.run(hidden_layer_1['weights']))

    correct = tf.equal(tf.argmax(output,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float32'))

    #print(test_data.shape,test_labels.shape)   
    print('Test Accuracy:',accuracy.eval({x:test_data,y:test_labels}))
    
    
