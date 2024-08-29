# Depression in SNN

This project can be accessed through codespace for testing. 

- Some general examples can be seen in the file `.\morris_lecar_sim.ipynb`.
- `Neuron` and `SNN` classes can be found under de folder `.\SNN`.
- The `.\Results` folder saves the results of training the SNN, both for a default model as well as depression.
- In `.\Modelling` a `Training` class can be seen, however this has been implemented in the `SNN` class directly. For future projects this class could be useful for better distribution of files.
- Finally, in `.\Emotion_data`, the data in can be found with both, input strings and output strings. 

## SNNs
In this folder, two classes are implemented: 

1. `Morris_Lecar_Neuron.py` applies a Morris-Lecar model of a neuron. It is structured so that it already stores a number of neurons in itself, for better computing.
2. `Default_SNN.py` applies the Morris-Lecar neuron (or rather neuron collection) to the process of training. All results are stored in the `.\Results` folder.

## Results
In this folder, the following can be found: 

1. File to the weights of the trained model.
2. Folder with specific records of training, in this case, `loss_history` and `accuracy_history`.
