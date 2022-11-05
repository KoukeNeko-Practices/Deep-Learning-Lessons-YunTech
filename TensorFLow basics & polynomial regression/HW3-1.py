import tensorflow as tf

# set global random seed
tf.random.set_seed(10)

# output 20 random valus range -5 ~ 5 from a uniform distribution
train = tf.random.uniform([20], -5, 5)
print(train)

# normalize the data 
train = (train - tf.reduce_min(train)) / (tf.reduce_max(train) - tf.reduce_min(train))
print(train)

# create our label
label = tf.greater(train, 0) #Returns the truth value of (x > y) element-wise.
label = tf.cast(train, tf.int64)   #Cast a tensor to a new type.
print(label)


# With batch size of 4, batch the transformed dataset; then shuffle it.
# Repeat the batched dataset so that each batch will be trained "twice".
# and create a data set called "ds" with Joint TensorFlow Dataset API
ds = tf.data.Dataset.from_tensor_slices((train, label)).batch(4).shuffle(10).repeat(2)

# print completed training dataset
for x, y in ds:
    print("batched_x", x)
    print("batched_y", y)

