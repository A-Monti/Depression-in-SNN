import numpy as np
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
    def __init__(self, input_size, hidden_size, output_size=1, model_name="NormalSNN") -> None:
        self.fc1_weights = np.random.randn(input_size, hidden_size)
        self.fc2_weights = np.random.randn(hidden_size, output_size)
        self.neuron = Morris_Lecar()
        self.model_name = model_name
        self.loss_history = []
        self.accuracy_history = []
        self.patience = 10

    def forward(self, x, dt=0.1):
        hidden_input = np.dot(x, self.fc1_weights)
        V, W = self.neuron.update(hidden_input, dt)
        output = np.dot(V, self.fc2_weights)
        return output
    
    def backward(self, x, y, output, learning_rate=0.01):
        # Calculate the gradient of the loss with respect to the output
        output_error = output - y

        # Calculate the gradient of the loss with respect to the second layer weights
        dW2 = np.dot(self.neuron.V.T, output_error)
        
        # Calculate the gradient of the loss with respect to the hidden layer
        hidden_error = np.dot(output_error, self.fc2_weights.T)
        
        # Calculate the gradient of the loss with respect to the first layer weights
        dW1 = np.dot(x.reshape(-1, 1), hidden_error.reshape(1, -1))
        
        # Update the weights
        self.fc1_weights -= learning_rate * dW1
        self.fc2_weights -= learning_rate * dW2

    def saveAnalytics(self) -> None:
        """
        Save results of training 
        """
        np.save(f"Results/{self.model_name}/loss_history.npy", self.loss_history)
        np.save(f"Results/{self.model_name}/accuracy_history.npy", self.accuracy_history)

    def train(self, X_train, y_train, X_val, y_val, epochs=100, learning_rate=0.01, batch_size=32):
        best_loss = float("inf")
        patience_counter = 0

        y_val = np.eye(len(np.unique(y_val)))[y_val]

        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X = X_train[indices]
            y = y_train[indices]

            epoch_loss = 0
            correct_prediction = 0

        # Process data in batches
        # for start in range(0, len(X), batch_size):
        #     end = start + batch_size
        #     batch_X = X[start:end]
        #     batch_y = y[start:end]

            batch_loss = 0
            batch_correct_prediction = 0

            for x, y in zip(X_train, y_train):
                # Forward pass
                output = self.forward(x)

                # Backward pass and weight update
                self.backward(x, y, output, learning_rate)

                # Calculate loss
                loss = np.mean((output - y) ** 2)
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
            # if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {average_loss}, Accuracy: {accuracy}')

        self.saveAnalytics()