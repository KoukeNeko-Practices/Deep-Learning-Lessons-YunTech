import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x_train = np.arange(10).reshape((10, 1))
y_train = 2 * x_train ** 2 + 3 * x_train + 1

# Normalize the data
x_train = (x_train - np.mean(x_train)) / np.std(x_train)

# join x_train and y_train
ds = tf.data.Dataset.from_tensor_slices((tf.cast(x_train, tf.float32), tf.cast(y_train, tf.float32))).batch(1).shuffle(buffer_size=len(y_train)).repeat(count=None)
    
# create a model with MyModel class for polynomial regression
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w1 = tf.Variable(0.0, name="weight1")
        self.w2 = tf.Variable(0.0, name="weight2")
        self.b = tf.Variable(0.0, name="bias")

    def call(self, x):
        return self.w1 * x ** 2 + self.w2 * x + self.b
    
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW1, dW2, db = tape.gradient(current_loss, [model.w1, model.w2, model.b])
    model.w1.assign_sub(learning_rate * dW1)
    model.w2.assign_sub(learning_rate * dW2)
    model.b.assign_sub(learning_rate * db)

model = MyModel()
model.build(input_shape=(None, 1))
model.summary()

tf.random.set_seed(1)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
step_per_epoch = int(len(x_train) / batch_size)

w1s, w2s, bs = [], [], []

# with custom training function

for i, batch in enumerate(ds):
    if i >= step_per_epoch * num_epochs:
        break
    w1s.append(model.w1.numpy())  
    w2s.append(model.w2.numpy())  
    bs.append(model.b.numpy())
    
    bx, by = batch
    loss = loss_fn(model(bx), by)
    
    train(model, bx, by, learning_rate)
    if i % log_steps == 0:
        print("Epoch: %d, Step: %d, loss: %f" % (i // step_per_epoch, i, loss.numpy()))
    
print("Final Parameters: w1 = %f, w2 = %f, b = %f" % (model.w1.numpy(), model.w2.numpy(), model.b.numpy()))

x_test = np.linspace(0, 9, 100).reshape((-1, 1))
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
y_pred = model(tf.cast(x_test, tf.float32))

fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(2, 2, 1)
plt.plot(x_train, y_train, 'o', label='Training Data', markersize=10)
plt.plot(x_test, y_pred, '--', label='Fitted line', linewidth=3)
plt.legend()
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(2, 2, 2)
plt.plot(w1s, label='w1', linewidth=3)
plt.plot(w2s, label='w2', linewidth=3)
plt.plot(bs, label='b', linewidth=3)
plt.legend()
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Parameter Value', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

plt.suptitle('Polynomial Regression', fontsize=20)


# with Keras API
model = MyModel()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), loss='mse')
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)

# plot the results
ax = fig.add_subplot(2, 2, 3)
plt.plot(x_train, y_train, 'o', label='Training Data', markersize=10)
plt.plot(x_test, model.predict(x_test), '--', label='Fitted line', linewidth=3)
plt.legend()
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
ax = fig.add_subplot(2, 2, 4)
plt.plot(history.history['loss'], label='loss', linewidth=3)
plt.legend()
ax.set_xlabel('Epoch', size=15)
ax.set_ylabel('Loss', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)


plt.show()

    
    

    