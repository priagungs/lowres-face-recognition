import tensorflow as tf
import numpy as np
import time

class Siamese():

  def __init__(self, input_shape=(128, )):
    self.input_shape = input_shape

  def load_dataset(self, dataset_path):
    data = np.load(dataset_path)
    self.x_train = data["arr_0"]
    self.y_train = data["arr_1"]
    self.x_test = data["arr_2"]
    self.y_test = data["arr_3"]
    self.input_shape = (self.x_train.shape[2], )
    print(self.input_shape)
    print(self.x_train.shape)

  def compile_model(self):
    input_1 = tf.keras.Input(shape=self.input_shape)
    input_2 = tf.keras.Input(shape=self.input_shape)

    l1_layer = tf.keras.layers.Lambda(lambda tensors: tf.keras.backend.abs([tensors[0] - tensors[1]]))
    l1_distance = l1_layer([input_1, input_2])

    x = tf.keras.layers.Dense(100, activation="relu")(l1_distance)
    x = tf.keras.layers.Dropout(0.8)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    self.model = tf.keras.Model([input_1, input_2], out)
    self.model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")

  def train(self, epochs=2400, batch_size=512):
    self.compile_model()

    log_dir = "logs/fit/facenet/" + time.strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_path = "models/classifier/facenet/optimized/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True)

    self.model.fit(
      [self.x_train[:, 0, :], self.x_train[:, 1, :]], 
      self.y_train,
      validation_split=0.2,
      epochs=epochs, 
      batch_size=batch_size, 
      callbacks=[cp_callback, tensorboard_callback]
    )

  def test(self):
    print(self.x_test[:, 0, :].shape)
    self.model.evaluate([self.x_test[:, 0, :], self.x_test[:, 1, :]], self.y_test, batch_size=128)

  def load_model(self, path):
    self.compile_model()
    self.model.load_weights(path)

  def predict(self, x):
    return self.model.predict(x)[0][0][0]
  
  def get_model(self):
    return self.model

if __name__ == "__main__":
  siamese = Siamese()
  siamese.load_dataset("datasets/lfw-pair-dataset_facenet.npz")
  # siamese.load_dataset("datasets/lfw-pair-test-facenet.npz")
  # siamese.load_model("models/classifier/facenet/new/cp.ckpt")
  # siamese.get_model().summary()
  siamese.train()
  siamese.test()

