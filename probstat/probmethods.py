import numpy as np
from scipy.stats import skew
from typing import Tuple
import matplotlib.pyplot as plt

def bernoulliDist(prob: float, size: int, exp: int, plot: bool)->Tuple[float,float,float]:
  '''
  Bernoulli distribution of a discrete random variable
  Arguments:
    prob (p): probability of success (i.e., 0 < prob < 1)
    size    : number of simulations per experiment
    exp     : number of experiments
    plot    : make various plots
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # ----- Simulation -----
  # NumPy array with values given a probability
  vals = np.random.choice([0,1], p=[1-prob,prob], size=(exp,size))
  # Bernoulli simulated stats for each experiment
  meanArr = np.mean(vals,axis=1)
  varArr = np.std(vals,axis=1)**2.0
  skArr = skew(vals,axis=1,bias=False)
  # Bernoulli simulated stats average
  meanSim = np.mean(meanArr)
  varSim = np.mean(varArr)
  skSim = np.mean(skArr)
  # ----- Theory -----
  # Expected value
  # EV = 1*p + 0*(1-p) = p
  meanTh = prob
  # Variance
  # v = p*q = p*(1-p)
  varTh = prob*(1-prob)
  # Skewness
  # s = (1-2p)/sqrt(p*q)
  skTh = (1-2*prob)/np.sqrt(prob*(1-prob))
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated mean is {skSim:.5f} (expected {skTh:.5f})")
  # ----- Plot -----
  if plot:
    # Distribution (up to 7 experiments)
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Distribution (p={prob}, simulations={exp,size})")
    bins = np.arange(0,2.1,1)
    plt.xticks(bins)
    for i in range(exp):
      if i > 6: break
      success = np.sum(vals[i])
      plt.hist(vals[i],bins=bins, histtype='step', label=f"Experiment {i+1} ({success})")
    expsuccess = prob*size
    plt.axhline(y=expsuccess,color="grey",linestyle="dashed",label=f"Expected ({expsuccess:.2f})")
    plt.legend(title="Successes")
    plt.xlabel("Failure/Success")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDist.pdf")
    # Mean
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Means (p={prob}, simulations={exp,size})")
    plt.hist(meanArr,label=f"Mean: {meanSim:.5f}")
    plt.axvline(x=meanTh,color="grey",linestyle="dashed",label=f"Expected ({meanTh:.5f})")
    plt.legend()
    plt.xlabel("Indep. Distribution Mean")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDistMeans.pdf")
    # Variance
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Variances (p={prob}, simulations={exp,size})")
    plt.hist(varArr,label=f"Variance: {varSim:.5f}")
    plt.axvline(x=varTh,color="grey",linestyle="dashed",label=f"Expected ({varTh:.5f})")
    plt.legend()
    plt.xlabel("Indep. Distribution Variance")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDistVars.pdf")
    # Skewness
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Skewnesses (p={prob}, simulations={exp,size})")
    plt.hist(skArr,label=f"Skewness: {skSim:.5f}")
    plt.axvline(x=skTh,color="grey",linestyle="dashed",label=f"Expected ({skTh:.5f})")
    plt.legend()
    plt.xlabel("Indep. Distribution Skewness")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDistSkews.pdf")
  return meanSim,varSim,skSim

def bernoulliDistTheoryVar()->None:
  '''
  Bernoulli distribution simulated variance
  Arguments:
    None
  Returns:
    None
  '''
  # Variables
  binwidth = 1e-2
  size=1000
  # Array of probabilities to simulate
  probs = np.arange(0,1+binwidth,binwidth)
  variances = []
  # Simulated values
  for prob in probs:
    vals = np.random.choice([0,1],p=[prob,1-prob],size=size)
    varSim = np.std(vals)**2
    variances.append(varSim)
  # Expected values
  varTh = probs*(1-probs)
  # Plot
  plt.figure(figsize=(8,6))
  plt.title(f"Bernoulli Distribution Theory Variance")
  plt.plot(probs,variances,marker=".",linestyle='None',label=f"Simulations ({size})")
  plt.plot(probs,varTh,label="Theory")
  plt.legend()
  plt.savefig("bernoulliDistTheoryVars.pdf")
  return
  
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

if __name__ == "__main__":
  bernoulliDist(0.1, 1000, 1000, True)
  bernoulliDistTheoryVar()
