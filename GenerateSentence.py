import sys
import numpy as np
from keras.models import model_from_yaml
from random import randint

class GenerateSentence(object):

    def __init__(self):
    
        # Open and read data from the file
        self.data = open('data/sonnets.txt', 'r').read()
        self.chars = sorted(list(set(self.data)))
        self.data_size, self.vocab_size = len(self.data), len(self.chars)

        # Create dictionaries mapping individual characters to indices and vice versa
        self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
        self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }
        self.maxLen = 50

        # Build the network from loaded architecture and weights
        self.architecture = ''
        with open('model_architecture.yaml', 'r') as file:
            self.architecture = file.read()

        self.model = model_from_yaml(self.architecture)
        self.model.load_weights('weights.hdf5') 
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    def sample(self, preds, temperature=1.0):
        
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_output(self):
 
        generated = ''
        usr_input = raw_input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
        sentence = ('{0:0>' + str(self.maxLen) + '}').format(usr_input).lower()
        generated += usr_input 

        sys.stdout.write("\n\nHere is your poem: \n\n") 
        sys.stdout.write(usr_input)
        
        for i in range(500):

            x_pred = np.zeros((1, self.maxLen, self.vocab_size))

            for t, char in enumerate(sentence):
                if char != '0':
                    x_pred[0, t, self.char_to_ix[char]] = 1

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = self.sample(preds, temperature = 1.0)
            next_char = self.ix_to_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()

            if next_char == '\n':
                continue
 
if __name__ == '__main__':
    
    sol = GenerateSentence()
    sol.generate_output()
