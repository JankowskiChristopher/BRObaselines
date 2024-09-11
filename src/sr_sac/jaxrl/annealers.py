import numpy as np
 
class ExponentialDecreasingMultiplier():
    def __init__(self, init, final, timesteps, return_int=False):
        self.initval = 1.05
        self.finalval = 0.05
        self.timesteps = timesteps
        self.start = np.log(self.initval)
        self.end = np.log(self.finalval)
        self.anneal_step = 0
        self.offset = final
        self.multiplier = init - final
        self.return_int = return_int
        
    def step(self):
        steps_left = self.timesteps - self.anneal_step
        bonus_frac = steps_left / self.timesteps
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (self.start - self.end) + self.end
        new_value = np.exp(new_value)
        self.anneal_step += 1
        if self.return_int:
            return int(np.round(self.rescale(np.clip((new_value - 0.05), 0.0, 1.0)), 0))
        else:
            return float(self.rescale(np.clip((new_value - 0.05), 0.0, 1.0)))

    def rescale(self, value):
        rescaled_value = value * self.multiplier + self.offset
        return rescaled_value
    
class ExponentialIncreasingMultiplier():
    def __init__(self, init, final, timesteps):
        self.initval = 0.05
        self.finalval = 1.05
        self.timesteps = timesteps
        self.start = np.log(self.initval)
        self.end = np.log(self.finalval)
        self.anneal_step = 0
        self.offset = init
        self.multiplier = final
        
    def step(self):
        steps_left = self.timesteps - self.anneal_step
        bonus_frac = steps_left / self.timesteps
        bonus = np.clip(bonus_frac, 0.0, 1.0)
        new_value = bonus * (self.start - self.end) + self.end
        new_value = np.exp(new_value)
        self.anneal_step += 1
        return float(self.rescale(np.clip((new_value - 0.05), 0.0, 1.0)))
    
    def rescale(self, value):
        rescaled_value = self.multiplier * value + (1 - value) * self.offset
        return rescaled_value    
    

'''
anneal = ExponentialDecreasingMultiplier(10, 1, 10000, True)

vallist = []
for i in range(50000):
    val = anneal.step()
    vallist.append(val)

import matplotlib.pyplot as plt
vallist = np.array(vallist)
plt.plot(np.arange(50000), vallist)
plt.show()
'''