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
    def __init__(self, input_size, hidden_size, output_size) -> None:
        self.fc1_weights = np.random.randn(input_size, hidden_size)
        self.fc2_weights = np.random.randn(hidden_size, output_size)
        self.neuron = Morris_Lecar()

    def forward(self, x, dt=0.1):
        hidden_input = np.dot(x, self.fc1_weights)
        V, W = self.neuron.update(hidden_input, dt)
        output = np.dot(V, self.fc2_weights)
        return output