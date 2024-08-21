import numpy as np

class Morris_Lecar:
    def __init__(self) -> None:
        self.V1 = -1.2
        self.V2 = 18
        self.V3 = 12
        self.V4 = 17.4
        self.phi = 0.04 
        self.gCa = 4.4 # Conductance of Calcium ions
        self.gK = 8 # Conductance of Potassium ions
        self.gL = 2 # Leak conductance
        self.VCa = 120 # Reversal conductance for Calcium ions
        self.VK = -84 # Reversal conductance for Potassium ions
        self.VL = -60 # Leak reversal potential
        self.I_ext = 40 # External current
        self.V = -60.0 # Membrane potential
        self.W = 0.0 # Recovery Variable
        self.V_vals = []

    def m_inf(self, V: float) -> float:
        """
        Steady-state value for activation variables, dependant on membrane potential `V`.
        
        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Calculation of activation
        """
        return 0.5 * (1 + np.tanh((V - self.V1) / self.V2))
    
    
    def w_inf(self, V: float) -> float:
        """
        Steady-state value for recovery variables, dependant on membrane potential `V`.
        
        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Calculation of recovery
        """
        return 0.5 * (1 + np.tanh((V - self.V3) / self.V4))


    def tau_w(self, V: float) -> float:
        """
        Time constant for recovery variable `W`, dependant on membrane potential `V`.
        
        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Calculation of recovery
        """
        return 1 / (self.phi * np.cosh((V - self.V3) / (2 * self.V4)))
    

    def I_Ca(self, V: float) -> float:
        """
        Calcium variables calculator for easier use in update. 

        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Calculation of Calcium
        """
        return self.gCa * self.m_inf(V) * (V - self.VCa)
    

    def I_K(self, V: float) -> float:
        """
        Potassium variables calculator for easier use in update. 

        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Calculation of Potassium
        """
        return self.gK * self.W * (V - self.VK)
    

    def I_L(self, V: float) -> float:
        """
        Potassium variables calculator for easier use in update. 

        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Calculation of Potassium
        """
        return self.gL * (V - self.VL)
    

    def dV(self, I: float, V: float, W: float) -> float:
        """
        Function for changes in membrane potential. Calculations are based on Calcium Potassium 

        Attributes: 
        @float - V: Current membrane potential

        Returns: 
        @float: Change of membrane potential
        """
        # return self.I_ext - self.I_Ca(V) - self.I_K(V) - self.I_L(V)
        return I - self.I_Ca(V) - self.I_K(V) - self.I_L(V)
    

    def dW(self, V: float, W: float) -> float:
        """
        Function for changes in Recovery variable. Calculations are based on Calcium Potassium 

        Attributes: 
        @float - V: Current Recovery variable

        Returns: 
        @float: Calculation of recovery variable
        """
        return (self.w_inf(V) - W) / self.tau_w(V)
        # return self.phi * (self.beta_m(V) - W)
    
    def update(self, I, dt):
        """
        Function to update the membrane potential 
        """
        self.V = self.V + dt * self.dV(I, self.V, self.W)
        self.W = self.W + dt * self.dW(self.V, self.W)
        self.V_vals.append(self.V)
        return self.V, self.W

    def STDP(self, delta_t, A_plus=0.005, A_minus=0.005, t_plus=20.0, t_minus=20.0):
        if delta_t > 0:
            weight_change = A_plus * np.exp(-delta_t / t_plus)
            print(f"Weight change due to STDP: {weight_change}.")
            return weight_change
        else:
            weight_change = -A_minus * np.exp(delta_t / t_minus)
            print(f"Weight change due to STDP: {weight_change}.")
            return weight_change
        
    # def plot(self, time):
    #     # Plotting the results
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(time, self.V, label='Membrane potential (V)')
    #     plt.xlabel('Time (ms)')
    #     plt.ylabel('Membrane potential (mV)')
    #     plt.title('Morris-Lecar Model')
    #     plt.legend()
    #     plt.show()
        