import numpy as np

class Train_Model:
    def __init__(self, model, X_train, y_train, epochs=10, learning_rate=0.01, model_name="NormalSNN"):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.loss_history = []
        self.accuracy_history = []
        self.spike_record = []
        self.membrane_potential = []

    def saveAnalytics(self) -> None:
        """
        Save results of training 
        """
        np.save(f"Results/{self.model_name}/loss_history.npy", self.loss_history)
        np.save(f"Results/{self.model_name}/accuracy_history.npy", self.accuracy_history)
        # np.save(f"Results/{self.model_name}/spike_record.npy", self.spike_record)
        # np.save(f"Results/{self.model_name}/membrane_potential.npy", self.membrane_potential)

    def saveModel(self):
        np.savez(f"Results/{self.model_name}.npz", fc1_weights=self.model.fc1_weights, fc2_weights=self.model.fc2_weights)
        self.saveAnalytics()

    def getAnalytics(self):
        return self.loss_history, self.accuracy_history, self.spike_record, self.membrane_potential
    def train(self):
        """
        Training Function
        """
        for epoch in range(self.epochs):
            epoch_loss = 0
            correct_predictions = 0

            for x, y in zip(self.X_train, self.y_train):
                # Forward pass
                output = self.model.forward(x)
                
                # Ensure y is an array
                if np.isscalar(y):
                    y = np.array([y])
                
                # Compute loss
                loss = np.mean((output - y) ** 2)
                epoch_loss += loss
                
                # Calculate accuracy
                predicted_class = np.argmax(output)
                true_class = np.argmax(y)
                if predicted_class == true_class:
                    correct_predictions += 1
                
                # Record spike and membrane potential
                self.spike_record.append(self.model.neuron.V_vals[-1])  # Assuming last value is the spike
                self.membrane_potential.append(self.model.neuron.V)
                
                # Backward pass (gradient descent)
                grad_output = 2 * (output - y) / len(y)
                grad_fc2_weights = np.outer(self.model.neuron.V, grad_output)
                grad_hidden = np.dot(grad_output, self.model.fc2_weights.T)
                grad_fc1_weights = np.outer(x, grad_hidden)
                
                # Update weights
                self.model.fc2_weights -= self.learning_rate * grad_fc2_weights
                self.model.fc1_weights -= self.learning_rate * grad_fc1_weights

            # Calculate average loss and accuracy for the epoch
            avg_loss = epoch_loss / len(self.X_train)
            accuracy = correct_predictions / len(self.X_train)
            
            # Append to history lists
            self.loss_history.append(avg_loss)
            self.accuracy_history.append(accuracy)

            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss}, Accuracy: {accuracy}')

        self.saveModel()
        # return self.loss_history, self.accuracy_history, self.spike_record, self.membrane_potential
