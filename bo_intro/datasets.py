import torch
import numpy as np
import math


class Sine:
    def __init__(self, config):
        bounds = torch.tensor([[1.0, 2.0]], dtype=torch.double)
        self.min, _ = torch.min(bounds, dim=1, keepdim=True)
        self.min = torch.transpose(self.min, 0, 1)
        self.interval = torch.abs(bounds[:, 0] - bounds[:, 1])
        self.noise = config['noise']
        self.dim = bounds.shape[0]
        self.num_points = config['initial_observations']
        self.x = torch.rand(self.num_points, self.dim, dtype=torch.double)
        self.y = self.query(self.x)
        self.max = 1

    def query(self, x):
        x_rescaled = self.rescale(x)
        y = torch.sin(x_rescaled * (2 * math.pi))
        y = self.add_noise(y)
        return y
        
    def add(self, new_x, new_y):
        self.x = torch.cat([self.x, new_x])
        self.y = torch.cat([self.y, new_y])

    def add_noise(self, y):
        if self.noise > 0:
            y += torch.randn(y.size(), dtype=torch.double) * self.noise
        if y.dim()==1:
            y = y.unsqueeze(1)
        return y

    def scale(self, x):
        """
        scale from real world interval to unit cube
        """
        x_scaled = (x - self.min)/self.interval
        return x_scaled

    def rescale(self, x):
        """
        scale unit cube to real world interval
        """
        x_rescaled = x * self.interval + self.min
        return x_rescaled

class Branin(Sine):
    """
    negative of the branin function https://www.sfu.ca/~ssurjano/branin.html
    """
    def __init__(self, config):
        bounds = torch.tensor([[-5.0, 10.0], [0.0, 15.0]], dtype = torch.double)
        self.min, _ = torch.min(bounds, dim=1, keepdim=True)
        self.min = torch.transpose(self.min, 0, 1)
        self.interval = torch.abs(bounds[:, 0] - bounds[:, 1])
        self.noise = config['noise']
        self.dim = bounds.shape[0]
        self.num_points = config['initial_observations']
        self.x = torch.rand(self.num_points, self.dim, dtype=torch.double)
        self.y = self.query(self.x)
        self.max = -0.397887

    def query(self, x):
        x = self.rescale(x)
        b = 5.1 / (4*math.pi**2)
        c = 5 / math.pi
        t = 1/ (8*math.pi)
        y = (x[:, 1] - b * x[:, 0] **2 + c * x[:, 0] - 6) **2 +\
             10 * (1 - t) * torch.cos(x[:, 1]) + 10
        y*=-1
        y = self.add_noise(y)
        return y