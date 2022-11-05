import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 根據方程式𝑦=2𝑥2+3𝑥+1，我們產生10筆樣本的資料集
x_traing = np.arange(10).reshape(10, 1)
y_train = 2 * x_traing ** 2 + 3 * x_traing + 1

x_test = np.linspace(0, 9, 100).reshape(-1, 1)
y_test = 2 * x_test ** 2 + 3 * x_test + 1

# create a subplots
# fig = plt.figure()
plt.subplots(1, 2, figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x_traing, y_train, 'ro', label='Training data')
plt.subplot(1, 2, 2)
plt.plot(x_test, y_test, 'bo', label='Testing data')
plt.show()

# 樣本集的特徵X_train經過以下之標準化的處理
x_train_norm = (x_traing - x_traing.mean()) / x_traing.std()
x_test_norm = (x_test - x_test.mean()) / x_test.std()

# 請使用資料集{X_train_norm, y_train}來訓練此非線性回歸模型，求出該模型參數𝑤2,𝑤1,𝑏的數值


# Epoch的次數為200，batch size的大小為1

epochs = 200
batch = 1
lr = 0.01

# 建立模型
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(1, input_shape=(1,), activation='linear', use_bias=True)
    ]
)

# compile model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss='mse', metrics=['mse'])

# fit the model
model.fit(x_train_norm, y_train, epochs=epochs, batch_size=batch, verbose=1)

# # define a optimizer
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')


# # use tf.keras.Sequential to build a model for linear regression
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(1, use_bias=True, input_shape=(1,))
# ])

# # Adam optimizer
# optimizer = tf.keras.optimizers.Adam(
#     learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-05, amsgrad=False,
#     name='Adam')
  
# # Model compiling settings
# model.compile(loss='mse', optimizer=optimizer, metrics=['mae','mse'])

# # A mechanism that stops training if the validation loss is not improving for more than n_idle_epochs.
# n_idle_epochs = 100
# # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=n_idle_epochs, min_delta=0.01)

# # Training loop
# n_epochs = epochs
# history = model.fit(
#   x_train_norm, y_train, batch_size=batch,
#   epochs=n_epochs, validation_split = 0.1, verbose=2,)


# print('keys:', history.history.keys())

# import numpy as np
# import pandas as pd
# import seaborn as sns
# # Returning the desired values for plotting and turn to numpy array
# mae = np.asarray(history.history['mae'])
# val_mae = np.asarray(history.history['val_mae'])
# # Creating the data frame
# num_values = (len(mae))
# values = np.zeros((num_values,2), dtype=float)
# values[:,0] = mae
# values[:,1] = val_mae
# # Using pandas to frame the data
# steps = pd.RangeIndex(start=0,stop=num_values)
# data = pd.DataFrame(values, steps, columns=["mae", "va-mae"])
# # Plotting
# sns.set(style="whitegrid")
# sns.lineplot(data=data, palette="tab10", linewidth=2.5)
# sns.despine()
# # model.summary()

# # # compile the model and fit the data
# # model.compile(optimizer=optimizer, loss="mse", metrics=["accuracy", "mse", "mae"])
# # model.fit(x_train_norm, y_train, epochs=epochs, batch_size=batch)

# # # print the model parameters
# # print(model.get_weights())

# # # predict the y value for the test data
# # print(model.predict(x_test_norm))