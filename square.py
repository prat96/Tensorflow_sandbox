from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

input_num = np.arange(-20, 20, 3)
square = np.square(input_num)
# print("Squared: ", square[0:10])

for i, c in enumerate(input_num):
    print("{} input = {} squared".format(c, square[i]))

l0 = tf.keras.layers.Dense(units=500, input_shape=[1], activation='relu')
# rl0 = tf.keras.layers.LeakyReLU()
l1 = tf.keras.layers.Dense(units=500, activation='relu')
# rl1 = tf.keras.layers.LeakyReLU()
d0 = tf.keras.layers.Dropout(0.4)
l2 = tf.keras.layers.Dense(units=100, activation='relu')
# rl2 = tf.keras.layers.LeakyReLU()
l3 = tf.keras.layers.Dense(units=100, activation='relu')
# rl3 = tf.keras.layers.LeakyReLU()
l4 = tf.keras.layers.Dense(units=100, activation='relu')
# rl4 = tf.keras.layers.LeakyReLU()
# l5 = tf.keras.layers.Dense(units=20, activation='relu')
# l6 = tf.keras.layers.Dense(units=500, activation='relu')

l7 = tf.keras.layers.Dense(units=1, activation='linear')
model = tf.keras.Sequential([l0, l1, l2, l3, l4, l7])

start = time.time()

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0001))

history = model.fit(input_num, square, epochs=1500, verbose=True, batch_size=10)
print("Finished training the model")

end = (time.time() - start)
print("\nTime taken: ", (end / 60), "mins")

print(model.summary())

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
print("\nModel", model.history)
plt.show()

# user_number = (input('Enter number:'))

prediction = model.predict([1.0])
print("\nSquare: ", int(prediction))

tf.keras.models.save_model(model, "./models/model.square", overwrite=True, include_optimizer=True)

# print("These are the layer variables: {}".format(l0.get_weights()))
# print("These are the layer variables: {}".format(l1.get_weights()))
# print("These are the layer variables: {}".format(l2.get_weights()))
