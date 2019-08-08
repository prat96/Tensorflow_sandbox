from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


celsius_q = np.array([-40, -10,  0,  8, 15, 22], dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72], dtype=float)

for i, c in enumerate(celsius_q):
  print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))


l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
# l1 = tf.keras.layers.Dense(units=4)
# l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0])

start = time.time()

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

print("\nModel", model)

history = model.fit(celsius_q, fahrenheit_a, epochs=1200, verbose=False)
print("Finished training the model")

end = (time.time() - start)
print("\nTime taken: ", end, "sec")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

# plt.show()

temperature = float(input("\nEnter Celsius: "))

prediction = model.predict([temperature])
print("\nFahrenheit: ", int(prediction))

print("These are the layer variables: {}".format(l0.get_weights()))
# print("These are the layer variables: {}".format(l1.get_weights()))
# print("These are the layer variables: {}".format(l2.get_weights()))