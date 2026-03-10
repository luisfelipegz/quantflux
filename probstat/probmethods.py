import numpy as np
from scipy.stats import skew
from typing import Tuple
import matplotlib.pyplot as plt

def centralLimitTheorem(simEvents: np.ndarray, meanExp: float, plot: bool)->None:
  '''
  The average of multiple large random experiments approximates a normal distribution.
  Expect the 68-95-99.7 rule to apply
  Arguments:
    simEvents : NxM NumPy array (N=number of experiments, M=events per experiment).
                Each entry corresponds to the output of each experiment
    meanExp   : Expected mean of distribution
    plot      : Plotting option
  Returns:
    None
    Just prints the simulated number of events within 1-3 standard deviations
  '''
  # Obtain means and std
  simulatedMeans = np.mean(simEvents, axis=1)
  simulatedStd = np.std(simulatedMeans)
  # Plot means distribution
  if plot:
    plt.figure(figsize=(8,6))
    plt.title("Simulated Means")
    plt.xlabel("Mean")
    plt.ylabel("Occurrences")
    plt.hist(simulatedMeans,bins=25)
    plt.axvline(meanExp, linestyle="-", color="grey", label=f"Expected Mean={meanExp:.2f} (sim. std={simulatedStd:.2f})")
    plt.legend(frameon=False)
    plt.savefig("cltMeans.pdf")
  # Get stats
  withinOneStd = np.abs(simulatedMeans-meanExp) <= 1*simulatedStd
  withinTwoStd = np.abs(simulatedMeans-meanExp) <= 2*simulatedStd
  withinThreeStd = np.abs(simulatedMeans-meanExp) <= 3*simulatedStd
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
