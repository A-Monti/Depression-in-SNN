import numpy as np
import os
from .Morris_Lecar_Neuron import Morris_Lecar

# class SNN:
#     def __init__(self, num_neurons, dt) -> None:
#         self.neurons = [Morris_Lecar() for _ in range(num_neurons)]
#         self.synaptic_weights = np.random.rand(num_neurons, num_neurons)
#         self.dt = dt
#         self.time = 0
#         self.spike_times = [[] for _ in range(num_neurons)]

#     def update(self):
#         for i, neuron in enumerate(self.neurons):
#             neuron.update(self.dt)
#             if neuron.V > 0:
#                 self.spike_times[i].append(self.time)
#                 for j in range(len(self.neurons)):
#                     if i != j:
#                         delta_t = self.time - self.spike_times[j][-1] if self.spike_times[j] else 0
#                         self.synaptic_weights[i,j] += neuron.STDP(delta_t)
#         self.time += self.dt

#     def run(self, duration):
#         steps = int(duration / self.dt)
#         for _ in range(steps):
#             self.update()


class SNN:
    def __init__(self, input_size, hidden_size, output_size=1, model_name="NormalSNN", depression=False) -> None:
        """
        model_name=["NormalSNN", "DepressionSNN"]
        """
        self.fc1_weights = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.fc2_weights = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.neuron = Morris_Lecar()
        if model_name == "DepressionSNN":
            self.directory = "Results/DepressionSNN"
        else: 
            self.directory = "Results/NormalSNN"
        self.loss_history = []
        self.accuracy_history = []
        self.patience = 10

        self.depression = depression

        # Synaptic depression variables
        self.synaptic_resources = np.ones_like(self.fc1_weights)  # Ensure same shape
        self.depression_rate = 0.1
        self.recovery_rate = 0.05

    def forward(self, x, dt=0.1):
        hidden_input = np.dot(x, self.fc1_weights)
        if self.depression: hidden_input = np.dot(x, self.fc1_weights * self.synaptic_resources)
        V, W = self.neuron.update(hidden_input, dt)
        output = np.dot(V, self.fc2_weights)
        return output
    
    def backward(self, x, y, output, learning_rate=0.01):
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
        """
        d_loss = self.directory + "/loss_history.npy"
        d_acc = self.directory + "/accuracy_history.npy"
        np.save(d_loss, self.loss_history)
        np.save(d_acc, self.accuracy_history)
    
    def saveModel(self):
        """
        Save model weights.
        """
        d = self.directory + ".npz"
        np.savez(d, fc1_weights=self.fc1_weights, fc2_weights=self.fc2_weights)
        self.saveAnalytics()

    def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01, batch_size=32):
            best_loss = float("inf")
            patience_counter = 0

            y_val = np.eye(len(np.unique(y_val)))[y_val]

            for epoch in range(epochs):
                # print(f"Starting epoch {epoch}")
                # Shuffle the data at the beginning of each epoch
                indices = np.arange(len(X_train))
                np.random.shuffle(indices)
                X = X_train[indices]
                y = y_train[indices].reshape(-1, 1)

                epoch_loss = 0
                correct_prediction = 0

                # Process data in batches
                for start in range(0, len(X), batch_size):
                    end = start + batch_size
                    batch_X = X[start:end]
                    batch_y = y[start:end]

                    batch_loss = 0
                    batch_correct_prediction = 0

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

                    epoch_loss += batch_loss
                    correct_prediction += batch_correct_prediction

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
                # if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {average_loss}, Accuracy: {accuracy}')

    # def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01, batch_size=32):
    #     best_loss = float("inf")
    #     patience_counter = 0

    #     y_val = np.eye(len(np.unique(y_val)))[y_val]

    #     for epoch in range(epochs):
    #         # Shuffle the data at the beginning of each epoch
    #         indices = np.arange(len(X_train))
    #         np.random.shuffle(indices)
    #         X = X_train[indices]
    #         y = y_train[indices]

    #         epoch_loss = 0
    #         correct_prediction = 0

    #         # Process data in batches
    #     # for start in range(0, len(X), batch_size):
    #     #     end = start + batch_size
    #     #     batch_X = X[start:end]
    #     #     batch_y = y[start:end]

    #         batch_loss = 0
    #         batch_correct_prediction = 0

    #         for x, y in zip(X_train, y_train):
    #             # Forward pass
    #             output = self.forward(x)

    #             # Backward pass and weight update
    #             self.backward(x, y, output, learning_rate)

    #             # Calculate loss
    #             loss = np.mean((output - y) ** 2)
    #             loss += 0.01 * np.sum(self.fc1_weights ** 2)
    #             batch_loss += loss

    #             # Calculate accuracy
    #             predicted_label = np.argmax(output)
    #             true_label = np.argmax(y)
    #             if predicted_label == true_label:
    #                 batch_correct_prediction += 1

    #         epoch_loss += batch_loss
    #         correct_prediction += batch_correct_prediction

    #         average_loss = epoch_loss / len(X)
    #         accuracy = correct_prediction / len(X)
    #         self.loss_history.append(average_loss)
    #         self.accuracy_history.append(accuracy)

    #         # Validation loss
    #         val_outputs = np.array([self.forward(x) for x in X_val])
    #         val_loss = np.mean((val_outputs - y_val) ** 2)

    #         # Early stopping check
    #         if val_loss < best_loss:
    #             best_loss = val_loss
    #             patience_counter = 0
    #         else:
    #             patience_counter += 1

    #         if patience_counter >= self.patience:
    #             print(f'Early stopping at epoch {epoch}')
    #             break

    #         # Print the loss and accuracy for monitoring
    #         # if epoch % 10 == 0:
    #         print(f'Epoch {epoch}, Loss: {average_loss}, Accuracy: {accuracy}')

    #     self.saveModel()