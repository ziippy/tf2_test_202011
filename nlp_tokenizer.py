import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = ['I love my dog',
             'I love my cat',
             'You love my dog!',
             'Do you think my dog is amazing?']

tokenizer = Tokenizer(num_words= 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
# {'my': 1, 'love': 2, 'dog': 3, 'i': 4, 'you': 5, 'cat': 6, 'do': 7, 'think': 8, 'is': 9, 'amazing': 10}


sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
# [[4, 2, 1, 3], [4, 2, 1, 6], [5, 2, 1, 3], [7, 5, 8, 1, 3, 9, 10]]