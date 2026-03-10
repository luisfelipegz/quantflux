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
