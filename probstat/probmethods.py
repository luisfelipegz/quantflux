import numpy as np
from scipy.stats import skew
from typing import Tuple
import matplotlib.pyplot as plt

def centralLimitTheorem(simEvents: np.ndarray, meanExp: float)->None:
  '''
  The average of multiple large random experiments approximates a normal distribution.
  Expect the 68-95-99.7 rule to apply
  Arguments:
    simEvents: NxM NumPy array (N=number of experiments, M=events per experiment).
               Each entry corresponds to the output of each experiment
    meanExp: Expected mean of distribution
  Returns:
    None
    Just prints the simulated number of events within 1-3 standard deviations
  '''
  simulatedMeans = np.mean(simEvents, axis=1)
  simulatedStd = np.std(simulatedMeans)
  print(simulatedMeans)
  print(simulatedStd)
  withinOneStd = np.abs(simulatedMeans-meanExp) <= 1*simulatedStd
  withinTwoStd = np.abs(simulatedMeans-meanExp) <= 2*simulatedStd
  withinThreeStd = np.abs(simulatedMeans-meanExp) <= 3*simulatedStd
  print(withinOneStd)
  print(f'Percentage of simulated means within one std of true mean: {np.mean(withinOneStd)*100:.2f}%')
  print(f'Percentage of simulated means within two std of true mean: {np.mean(withinTwoStd)*100:.2f}%')
  print(f'Percentage of simulated means within three std of true mean: {np.mean(withinThreeStd)*100:.2f}%')
  return

def oneDiceRoll(nRolls: int, nExp: int)->np.ndarray:
  '''
  Simulated rolls of one dice.
  Arguments:
    nRolls      : Number of rolls per experiment
    nExp        : Number of experiments
  Returns:
    np.ndarray  : nExp x nRolls array with dice output
  '''
  simulatedRolls = np.random.randint(low=1, high=7, size=(nExp,nRolls))
  return simulatedRolls

def twoDiceRollSum(nRolls: int, nExp: int)->np.ndarray:
  '''
  Simulated rolls of one dice.
  Arguments:
    nRolls      : Number of rolls per experiment
    nExp        : Number of experiments
  Returns:
    np.ndarray  : nExp x nRolls array with dice output
  '''
  simulatedRolls = np.sum(np.random.randint(low=1, high=7, size=(nExp,nRolls,2)),axis=2)
  return simulatedRolls

def fairDiceRollEV(size: int)->int:
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

if __name__=="__main__":
  nRolls = 500
  nExp = 1000
  rolls = twoDiceRollSum(nRolls,nExp)
  centralLimitTheorem(rolls,7)
