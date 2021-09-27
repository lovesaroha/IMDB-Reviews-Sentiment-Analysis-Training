# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model to detect sentiment in imdb movie review.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

# Get imdb subwords data.
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)
training , validation = imdb['train'], imdb['test']

# Parameters.
embedding_dim = 16
epochs = 10
batchSize = 128
BUFFER_SIZE = 10000

# Get tokenizer.
tokenizer = info.features['text'].encoder

# Create training data with labels.
training_data = training.shuffle(BUFFER_SIZE)
training_data = training_data.padded_batch(batchSize, tf.compat.v1.data.get_output_shapes(training_data))

# Create validation data with labels.
validation_data = validation.padded_batch(batchSize, tf.compat.v1.data.get_output_shapes(validation))

# Create model with 1 output unit for classification.
model = keras.Sequential([
    keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
    keras.layers.Bidirectional(keras.layers.GRU(32)),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

# Set loss function and optimizer.
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])


class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      # Stop when validation accuracy is more than 98%.
        if logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.98:
            print("\nTraining Stopped!")
            self.model.stop_training = True


# Callback function to check accuracy.
checkAccuracy = myCallback()

# Train model.
model.fit(training_data,epochs=epochs, callbacks=[
          checkAccuracy], batch_size=batchSize, validation_data=validation_data, verbose=1)
