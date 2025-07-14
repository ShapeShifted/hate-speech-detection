import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import keras

max_sequence_length = 46

def change_to_string(input):
    # obtain token list that's actually list instead of string
    from ast import literal_eval

    topic_words = input

    #for x in input:
    #    topic_words.append((literal_eval(x)))

    # change the list containing a list of elements into list of strings
    topic_strings = []

    for i in topic_words:
        temp = ' '.join(i)
        topic_strings.append(temp)

    topic_string = ' '.join(topic_strings)

    topic_string = [topic_string]

    return topic_string

def tokenization(input):
    
    # Load the tokenizer from the file
    with open('posts1352_standard_GRU_tokenizer_2025_2.pickle', 'rb') as tokenizer_path:
        loaded_tokenizer = pickle.load(tokenizer_path)

    sequences = loaded_tokenizer.texts_to_sequences(input)

    padded_data = pad_sequences(sequences, maxlen=max_sequence_length)

    return padded_data

def prediction(input):

    model = tf.keras.models.load_model('posts1352_standard_GRU_model_2025_2.keras')
    #model = tf.keras.models.load_model('posts1352_standard_GRU_model.h5',compile=False)
    #model.compile() 

    input = change_to_string(input)

    input = tokenization(input)

    predictions = model.predict(input)

    binary_predictions = (predictions > 0.5).astype(int)

    return binary_predictions[0][0]