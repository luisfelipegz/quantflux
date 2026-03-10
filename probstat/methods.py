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

def symmetricRandomWalk(steps: int, plot: bool)->np.ndarray:
  '''
  Random walk with 1/2 probability of going up by one or down by one
  Arguments:
    steps     : Number of steps to simulate
    plot      : Plotting option
  Returns:
    walk      : Simulated random walk
  '''
  choices = np.random.choice([1,-1],p=[1/2,1/2],size=steps)
  walk = np.cumsum(choices)
  walk = np.insert(walk,0,0)
  if plot:
    plt.figure(figsize=(8,6))
    plt.title("Symmetric Random Walk")
    plt.plot(walk)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.savefig("symmetricRandomWalk.pdf")
  return walk

def randomWalk(stepSizes: list[float], stepProbs: list[float], steps: int)->np.ndarray:
  '''
  Random walk with user defined step sizes and probabilities
  Arguments:
    stepSizes : List of size of possible steps
    stepProbs : List of probabilities for each possible step size
    steps     : Number of steps to simulate
    plot      : Plotting option
  Returns:
    walk      : Simulated random walk
  '''
  choices = np.random.choice(stepSizes,p=stepProbs,size=steps)
  walk = np.cumsum(choices)
  walk = np.insert(walk,0,0)
  if plot:
    plt.figure(figsize=(8,6))
    plt.title("Random Walk")
    plt.plot(walk)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.savefig("randomWalk.pdf")
  return walk
