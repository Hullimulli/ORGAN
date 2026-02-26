import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

class Memory:
    def __init__(self,num_elements):
        self.num_elements   = num_elements
        self.current_length = 0
        
        self.memory = None
        
    def add_and_return_element(self,element):
        if self.num_elements > self.current_length: 
            if self.current_length == 0:
                self.memory = element.detach()
            else:
                self.memory = torch.cat([self.memory,element.detach()],0)
            self.current_length += 1
        else:
            self.memory = torch.cat([self.memory[element.shape[0]:,...],element.detach()],0)
        
        return self.memory


class movingAverage:
    def __init__(self,N):
        self.values = np.zeros(N)
        self.it = 0

    def __call__(self, value = None):

        if value is not None:
            self.values[self.it % len(self.values)] = value
            self.it += 1

        return np.mean(self.values[:self.it])
        