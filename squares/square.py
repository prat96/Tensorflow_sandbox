from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
import time, os
import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

input_num = np.arange(-50, 50, 1)
square = np.square(input_num)


for i, c in enumerate(input_num):
    print("{} input = {} squared".format(c, square[i]))

model = tf.keras.Sequential(
    [tf.keras.layers.Dense(units=512, input_shape=[1], activation='relu'),
     tf.keras.layers.Dropout(0.4)  ,
     tf.keras.layers.Dense(units=1024, activation='relu'),
     tf.keras.layers.Dense(units=1, activation='linear')
     ])
start = time.time()

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.0005))

history = model.fit(input_num, square, epochs=1000, verbose=True, batch_size=64)
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

tf.keras.models.save_model(model, "./models/model.square_sgd", overwrite=True, include_optimizer=True)

# print("These are the layer variables: {}".format(l0.get_weights()))
# print("These are the layer variables: {}".format(l1.get_weights()))
# print("These are the layer variables: {}".format(l2.get_weights()))
