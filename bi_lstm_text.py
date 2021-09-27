# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Train keras model to detect sentiment in imdb movie review.
import numpy
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

# Get imdb text data.
imdb, info = tfds.load("imdb_reviews/plain_text", with_info=True, as_supervised=True)
training , validation = imdb['train'], imdb['test']

# Parameters.
token_size = 4000
embedding_dim = 16
text_length = 120
epochs = 20
batchSize = 256

# Prepare training data.
training_sentences = []
training_labels = []
for s,l in training:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())

# Prepare validation data.
validation_sentences = []
validation_labels = []
for s,l in validation:
  validation_sentences.append(str(s.numpy()))
  validation_labels.append(l.numpy())

# Create tokenizer.
tokenizer = Tokenizer(num_words=token_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Get text from word tokens.
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '') for i in text])

# Create training sequences.
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_data = numpy.array(pad_sequences(
    training_sequences, maxlen=text_length, padding="post", truncating="post"))
training_labels = numpy.array(training_labels)

# Create validation sequences.
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
validation_data = numpy.array(pad_sequences(
    validation_sequences, maxlen=text_length, padding="post", truncating="post"))
validation_labels = numpy.array(validation_labels)

# Create model with 1 output unit for classification.
model = keras.Sequential([
    keras.layers.Embedding(token_size, embedding_dim,
                           input_length=text_length),
    keras.layers.Bidirectional(keras.layers.LSTM(32)),
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
model.fit(training_data, training_labels, epochs=epochs, callbacks=[
          checkAccuracy], batch_size=batchSize, validation_data=(validation_data, validation_labels), verbose=1)


# Predict on a random validation text.
index = 7
text = validation_data[index]
prediction = model.predict(text.reshape(1, text_length, 1))

print("Prediciton : ", prediction[0][0] )
print("Label : " , validation_labels[index])
print("Text : ", decode_sentence(validation_data[index]))