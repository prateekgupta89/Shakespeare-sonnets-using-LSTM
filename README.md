# Shakespeare-sonnets-using-LSTM
This is a Keras implemenation of generating Shakespeare sonnets.
1) The training makes use of a set of shakespeare sonnets from which overlapping sentences of length 50 have been extracted to create training examples.
2) The LSTM model is a deep, stacked LSTM model (2 layer). The model has been trained for 100 epochs on a CPU. It can led to better results if trained for more epochs.
3) To train the model, use the command:
        python run.py
   This will save the model architecture in model.yaml and store the weights in a hdf5 file.
4) To generate sonnets, use the command:
        python GenerateSentence.py
   This will ask to input a shakespearean sentence of <= 50 characters in length and then use it to generate sonnets of 500 characters in length.
5) The implementation is based on the following implementation:
   https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

