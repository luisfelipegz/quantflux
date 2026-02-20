import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fairDiceRollEV(size: int):
  ''' 
  Expectation value of a fair dice roll.
  Arguments:
    size: Simulation size (i.e., number of rolls)
  Returns:
    avg: Average value of rolls (i.e. expectation value)
  '''
  # NumPy array with values
  vals = np.random.randint(low=1, high=7, size=size)
  # Compute the EV using the average
  avg = np.mean(vals)
  # Return probabilistic EV
  print(f"Average of {size} dice rolls is {avg}")
  return avg
