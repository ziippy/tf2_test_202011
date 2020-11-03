import tensorflow_datasets as tfds

# https://github.com/tensorflow/datasets/blob/master/docs/catalog/imdb_reviews.md
imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

tokenizer = info.features['text'].encoder
print(tokenizer.subwords)
# ['the_', ', ', '. ', 'a_', 'and_', 'of_', 'to_', 's_',  ...
# tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder

sample_string = 'TensorFlow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string is {}'.format(original_string))

# Tokenized string is [6307, 2327, 4043, 2120, 2, 48, 4249, 4429, 7, 2652, 8050]
# The original string is TensorFlow, from basics to mastery