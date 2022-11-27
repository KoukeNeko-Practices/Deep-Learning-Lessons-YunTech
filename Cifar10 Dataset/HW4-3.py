# %%
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt


BUFFER_SIZE = 128
BATCH_SIZE = 100
NUM_EPOCHS = 30
# %%
def preprocess(item):
    image = item["image"]
    label = item["label"]
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (-1,))  # reshape to (3072,)
    image = tf.math.divide(image, 255.0)
    return image, label[..., tf.newaxis]


# %%
datasets = tfds.load(name="cifar10")
cifar10_train = datasets["train"]


train_dataset = cifar10_train.map(preprocess)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
# %%
cifar10_test = datasets["test"]

test_dataset = cifar10_test.map(preprocess)
test_dataset = test_dataset.batch(BATCH_SIZE)

# %%
image_shape = next(iter(cifar10_train))["image"].shape
input_shape = np.prod(image_shape)

inputs = tf.keras.layers.Input(shape=(input_shape,))
hidden_1 = tf.keras.layers.Dense(100, activation="relu")(inputs)
outputs = tf.keras.layers.Dense(10, activation="softmax")(hidden_1)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.summary()


#%%
optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

hist = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=NUM_EPOCHS,
    verbose=1,
)
# %%
history = hist.history
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history["loss"], lw=4)
plt.plot(history["val_loss"], lw=4)
plt.legend(["Train loss", "Validation loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax.set_title("Model A", size=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(history["sparse_categorical_accuracy"], lw=4)
plt.plot(history["val_sparse_categorical_accuracy"], lw=4)
plt.legend(["Train Acc.", "Validation Acc."], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax.set_title("Model A", size=15)

plt.show()
# %%
inputs = tf.keras.layers.Input(shape=(input_shape,))
hidden = tf.keras.layers.Dense(5000, activation="relu")(inputs)
hidden = tf.keras.layers.Dense(4000, activation="relu")(hidden)
outputs = tf.keras.layers.Dense(10, activation="softmax")(hidden)
model2 = tf.keras.Model(inputs=inputs, outputs=outputs)
model2.summary()


#%%
optimizer = tf.keras.optimizers.SGD(lr=0.001, decay=1e-7, momentum=0.9)

model2.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

hist = model2.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=NUM_EPOCHS,
    verbose=1,
)
# %%
history = hist.history
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history["loss"], lw=4)
plt.plot(history["val_loss"], lw=4)
plt.legend(["Train loss", "Validation loss"], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax.set_title("Model B", size=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(history["sparse_categorical_accuracy"], lw=4)
plt.plot(history["val_sparse_categorical_accuracy"], lw=4)
plt.legend(["Train Acc.", "Validation Acc."], fontsize=15)
ax.set_xlabel("Epochs", size=15)
ax.set_title("Model B", size=15)

plt.show()