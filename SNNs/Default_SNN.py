import numpy as np
from .Morris_Lecar_Neuron import Morris_Lecar

class SNN:
    """
    This class will initiate a Morris_Lecar neuron, with weights, 
    train it and finally save the loss and accuracy of the model.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int=1, model_name: str="NormalSNN", depression: bool=False) -> None:
        """
        Initation function for class. 
        
        Attributes:
        @int - input_size: input size of model.
        @int - hidden_size: hidden size of model.
        @int - output_size: Default=1, output size for the model.
        @str - model_name: Possibility to create a NormalSNN or a DepressionSNN.
                        For easier access, default=NormalSNN.
        @bool - depression: Whether the model must have depression or not.
                        Default value = No depression

        Return:
        @None
        """
        # Weights initiation
        self.fc1_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.fc2_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)

        # Initiaze Neuron
        self.neuron = Morris_Lecar()

        # Directory to save the results of model
        if model_name == "DepressionSNN":
            self.directory = "Results/DepressionSNN"
        else: 
            self.directory = "Results/NormalSNN"

        # Save loss and average
        self.loss_history = []
        self.accuracy_history = []

        # Stop parameter
        self.patience = 10

        # Synaptic depression variables
        self.depression = depression
        self.synaptic_resources = np.ones_like(self.fc1_weights)  # Ensure same shape
        self.depression_rate = 0.1
        self.recovery_rate = 0.05

    def forward(self, x: np.array, dt: float=0.1) -> np.array:
        """
        Forward function for model training.

        Attributes: 
        @np.array - x: Current x to calculate forward.
        @float - dt: Current time. 

        Return:
        @np.array: Output after forward function.
        """
        hidden_input = np.dot(x, self.fc1_weights)

        # For depression
        if self.depression: hidden_input = np.dot(x, self.fc1_weights * self.synaptic_resources)

        # Update neurons
        V, W = self.neuron.update(hidden_input, dt)
        output = np.dot(V, self.fc2_weights)
        return output
    
    def backward(self, x: np.array, y: np.array, output: np.array, learning_rate: float=0.01) -> None:
        """
        Backward function for model training. 

        Attributes:
        @np.array - x: Current x to calculate backward.
        @np.array - y: Current y to calculate backward.
        @np.array - output: Output of forward function.
        @float - learning_rate=0.01: learning rate for model.

        Return:
        @None
        """
        # Calculate the gradient of the loss with respect to the output
        output_error = output - y / output.size

        # Calculate the gradient of the loss with respect to the second layer weights
        dW2 = np.dot(self.neuron.V.T, output_error)
        
        # Calculate the gradient of the loss with respect to the hidden layer
        hidden_error = np.dot(output_error, self.fc2_weights.T)
        
        # Calculate the gradient of the loss with respect to the first layer weights
        dW1 = np.dot(x.reshape(-1, 1), hidden_error.reshape(1, -1))
        
        # Ensure gradients are arrays
        if not isinstance(dW1, np.ndarray):
            dW1 = np.array(dW1)
        if not isinstance(dW2, np.ndarray):
            dW2 = np.array(dW2)

        # Gradient clipping
        dW1 = np.clip(dW1, -1, 1)
        dW2 = np.clip(dW2, -1, 1)

        # Update the weights
        self.fc1_weights -= learning_rate * dW1
        self.fc2_weights -= learning_rate * dW2

        # Apply synaptic depression
        if self.depression:
            self.synaptic_resources -= self.depression_rate * (self.synaptic_resources > 0)
            self.synaptic_resources += self.recovery_rate * (self.synaptic_resources < 1)

    def saveAnalytics(self) -> None:
        """
        Save results of training.

        Attributes: 
        @None

        Return: 
        @None

        """
        # Set directory
        d_loss = self.directory + "/loss_history.npy"
        d_acc = self.directory + "/accuracy_history.npy"

        # Save histories
        np.save(d_loss, self.loss_history)
        np.save(d_acc, self.accuracy_history)
    
    def saveModel(self):
        """
        Save model weights.

        Attributes: 
        @None

        Return: 
        @None

        """
        d = self.directory + ".npz"
        np.savez(d, fc1_weights=self.fc1_weights, fc2_weights=self.fc2_weights)
        self.saveAnalytics()

    def train(self, X_train: np.array, y_train: np.array, X_val: np.array, y_val: np.array, epochs: int=100, learning_rate: float=0.01, batch_size: int=32) -> None:
        """
        Training of model. 

        Attributes:
        @np.array - X_train: Data for training.
        @np.array - y_train: Output for training.
        @np.array - X_val: Data for validation.
        @np.array - y_val: Output for validation.
        @int - epochs: Epochs of training (Default=100).
        @float - learning_rate: Learning rate for model (Default=0.01).
        @int - batch_size: Batch Size for better training (Default=32).

        Return: 
        @None
        """
        # Variables set
        best_loss = float("inf")
        patience_counter = 0

        y_val = np.eye(len(np.unique(y_val)))[y_val]

        # Start of epoch training
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X = X_train[indices]
            y = y_train[indices].reshape(-1, 1)

            # Initiation Epoch variables
            epoch_loss = 0
            correct_prediction = 0

            # Process data in batches
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_X = X[start:end]
                batch_y = y[start:end]
                
                # Initiation Batch variables
                batch_loss = 0
                batch_correct_prediction = 0

                # Apply forward and backward functions
                for x, y in zip(batch_X, batch_y):
                    # Forward pass
                    output = self.forward(x)

                    # Backward pass and weight update
                    self.backward(x, y, output, learning_rate)

                    # Calculate loss
                    loss = np.mean((output - y) ** 2)
                    loss += 0.01 * np.sum(self.fc1_weights ** 2)
                    batch_loss += loss

                    # Calculate accuracy
                    predicted_label = np.argmax(output)
                    true_label = np.argmax(y)
                    if predicted_label == true_label:
                        batch_correct_prediction += 1

                # Variables update for epoch loss and prediction
                epoch_loss += batch_loss
                correct_prediction += batch_correct_prediction

            # Save history
            average_loss = epoch_loss / len(X)
            accuracy = correct_prediction / len(X)
            self.loss_history.append(average_loss)
            self.accuracy_history.append(accuracy)

            # Validation loss
            val_outputs = np.array([self.forward(x) for x in X_val])
            val_loss = np.mean((val_outputs - y_val) ** 2)

            # Early stopping check
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f'Early stopping at epoch {epoch}')
                break

            # Print the loss and accuracy for monitoring
            print(f'Epoch {epoch}, Loss: {average_loss}, Accuracy: {accuracy}')
            