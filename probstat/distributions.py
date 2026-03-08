import math
import numpy as np
from scipy.stats import skew
from typing import Tuple
import matplotlib.pyplot as plt

"""

  Bernoulli distribution:
    
    Discrete probability distribution of a random variable
     which takes the value 1 with probability p and the value 0 with probability q=1-p.
    
    Model for the set of possible outcomes of any single experiment that asks a yes-no question.
    
    i.e., represent a coin toss where 1 and 0 would represent heads and tails, respectively.

  Binomial distribution:
    
    Discrete probability distribution of the number of successes in a sequence of n independent 
     experiments, each asking a yes-no question, and each with its own outcome:
     success (with probability p) or failure (with probability q=1-p).

    Basis for the binomial test of statistical significance.

    Model the number of successes in a sample of size n drawn with replacement from a population 
     of size N.

    i.e., flipping a coin 10 times and counting the number of heads.

  Hypergeometric distribution:
  
    Discrete probability distribution that describes the probability of k successes in n draws
     without replacement from a finite population of size N that contains K objects with that
     feature, where in each draw is either a success or a failure.

    Basis for the hypergeometric distribution to measure statistical significance.

    i.e., drawing red marbels from an urn with red and green marbels without replacement.

  Geometric distribution:
  
    Discrete probability distribution of the number X of Bernoulli trials needed to get one success.
     Another way to put it, the number Y=X-1 of failures before the first success.

    Model the number of trials up to and including the first success.

    The geometric distribution is the only memoryless discrete probability distribution, where the
     number of previously failed trials does not affect the number of future trials needed for a
     success.

    i.e., probability a coin lands heads after 9 consecutive tails

  Uniform distribution:

    Continuous probability distribution that describes an experiment where there is an arbitrary
     outcome that lies between certain boundaries.

    The conitnuous uniform distribution with parameters a=0 and b=1 is called the standard uniform
     distribution

    i.e., maximum entropy probability distribution for a random variable X under no constraints.

  Normal distribution:

    Continuous probability distribution for a real-valued random variable. Often used to represent
     random variables whose distributions are not known.
    
    Their importance is partially due to the central limit theorem. It states that the average of
     many statistically independent samples of a random variable with finite mean and variance is
     itself a random variable whose distribution converges to a normal distribution as the number
     of samples increases.

    i.e. physical quantities that are expected to be the sum of many independent processes
    
"""

def bernoulliDist(prob: float, exp: int, size: int, plot: bool)->Tuple[float,float,float]:
  '''
  Bernoulli distribution of a discrete random variable
  Arguments:
    prob    : probability of success (i.e., 0 < prob < 1)
    exp     : number of experiments
    size    : number of simulations per experiment
    plot    : make various plots
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
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
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
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
    plt.legend(title="Successes",frameon=False)
    plt.xlabel("Failure/Success")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDist.pdf")
    # Mean
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Means (p={prob}, simulations={exp,size})")
    plt.hist(meanArr,label=f"Mean: {meanSim:.5f}")
    plt.axvline(x=meanTh,color="grey",linestyle="dashed",label=f"Expected ({meanTh:.5f})")
    plt.legend(frameon=False)
    plt.xlabel("Indep. Distribution Mean")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDistMeans.pdf")
    # Variance
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Variances (p={prob}, simulations={exp,size})")
    plt.hist(varArr,label=f"Variance: {varSim:.5f}")
    plt.axvline(x=varTh,color="grey",linestyle="dashed",label=f"Expected ({varTh:.5f})")
    plt.legend(frameon=False)
    plt.xlabel("Indep. Distribution Variance")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDistVars.pdf")
    # Skewness
    plt.figure(figsize=(8,6))
    plt.title(f"Bernoulli Skewnesses (p={prob}, simulations={exp,size})")
    plt.hist(skArr,label=f"Skewness: {skSim:.5f}")
    plt.axvline(x=skTh,color="grey",linestyle="dashed",label=f"Expected ({skTh:.5f})")
    plt.legend(frameon=False)
    plt.xlabel("Indep. Distribution Skewness")
    plt.ylabel("Num. Experiments")
    plt.savefig("bernoulliDistSkews.pdf")
  return meanSim,varSim,skSim

def bernoulliDistTheoryVar()->None:
  '''
  Bernoulli distribution variance
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
  plt.legend(frameon=False)
  plt.savefig("bernoulliDistTheoryVars.pdf")
  return

def binomialDist(prob: float, trials: int, exp: int, plot: bool)->Tuple[float,float,float]:
  '''
  Binomial distribution of a random variable
  Arguments:
    prob    : probability of success (i.e., 0 < prob < 1)
    trials  : number of trials per experiment (i.e., 40)
    exp     : number of experiments (i.e., 1000)
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # ----- Theory -----
  # Expected value
  # Each experiment is a Bernoulli trial
  # E[X] = E[x1]+E[x2]+...E[xn] = n*p
  meanTh = trials*prob
  # Variance
  # v = n*(p*q) = n*p*(1-p)
  varTh = trials*prob*(1-prob)
  # Skewness
  # s = (1-2p)/sqrt(p*q)
  skTh = (1-2*prob)/np.sqrt(trials*prob*(1-prob))
  # Probabilities
  #  Finding k successes with probability p in n trials is
  #   f(k,n,p) = (n choose k) p^k * q^(n-k)
  points = np.arange(0,trials+1.1,1)
  probsTh = [math.comb(trials,int(k))*prob**k*(1-prob)**(trials-int(k)) for k in points[:-1]]
  cumProbsTh = np.cumsum(probsTh)
  cumProbsTh = np.insert(cumProbsTh,0,0)
  # ----- Simulation -----
  # NumPy array with values given a probability
  vals = np.random.choice([0,1], p=[1-prob,prob], size=(exp,trials))
  # Calculate sum of successes
  sums = np.sum(vals, axis=1)
  # Binomial simulated stats
  meanSim = np.mean(sums)
  varSim = np.std(sums)**2.0
  skSim = skew(sums,bias=False)
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
  # ----- Plot -----
  if plot:
    # PMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,trials)
    plt.title(f"Binomial Distribution PMF (p={prob}, n={trials})")
    plt.hist(sums, bins=points, density=True, histtype="step", label=f"Simulation")
    plt.plot(probsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("X")
    plt.ylabel("P(X)")
    plt.legend(frameon=False)
    plt.savefig("binomialDistPMF.pdf")
    # CMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,trials)
    plt.title(f"Binomial Distribution CDF (p={prob}, n={trials})")
    plt.hist(sums, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation")
    plt.plot(cumProbsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("X")
    plt.ylabel("Cumulative Sum")
    plt.legend(frameon=False)
    plt.savefig("binomialDistCDF.pdf")
  return meanSim, varSim, skSim

def binomialDistRawSim(prob: float, trials: int, exp: int)->np.ndarray:
  '''
  Binomial distribution of a random variable
  Arguments:
    prob    : probability of success (i.e., 0 < prob < 1)
    trials  : number of trials per experiment (i.e., 40)
    exp     : number of experiments (i.e., 1000)
    plot    : option to plot
  Returns:
    sums    : numpy array with the sum of successes in each experiment
  '''
  # ----- Simulation -----
  # NumPy array with values given a probability
  vals = np.random.choice([0,1], p=[1-prob,prob], size=(exp,trials))
  # Convert to a probability mass function
  sums = np.sum(vals, axis=1)
  return sums

def binomialDists(probs: list, trials: int, exp: int, plot: bool)->Tuple[list, list, list]:
  '''
  Multiple binomial distributions of a random variable
  Arguments:
    probs   : list of probabilities of success (i.e., 0 < prob < 1)
    trials  : number of trials per experiment (i.e., 40)
    exp     : number of experiments (i.e., 1000)
    plot    : option to plot
  Returns:
    meansSim : list of simulated means
    varsSim  : list of simulated variances
    sksSim   : list of simulated skewnesses
  '''
  # Start plot
  if plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    axes[0].set_title(f"Binomial Distribution PMF")
    axes[0].set_xlim(0,trials)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("P(X)")
    axes[1].set_title(f"Binomial Distribution CDF")
    axes[1].set_xlim(0,trials)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Cumulative Sum")
  # Loop through each probability
  means = []
  variances = []
  skewnesses = []
  for p in range(len(probs)):
    print(f"Using prob {probs[p]}")
    # ----- Theory -----
    # Expected value
    # Each experiment is a Bernoulli trial
    # E[X] = E[x1]+E[x2]+...E[xn] = n*p
    meanTh = trials*probs[p]
    # Variance
    # v = n*(p*q) = n*p*(1-p)
    varTh = trials*probs[p]*(1-probs[p])
    # Skewness
    # s = (1-2p)/sqrt(p*q)
    skTh = (1-2*probs[p])/np.sqrt(trials*probs[p]*(1-probs[p]))
    # Probabilities
    #  Finding k successes with probability p in n trials is
    #   f(k,n,p) = (n choose k) p^k * q^(n-k)
    points = np.arange(0,trials+1.1,1)
    probsTh = [math.comb(trials,int(k))*probs[p]**k*(1-probs[p])**(trials-int(k)) for k in points[:-1]]
    cumProbsTh = np.cumsum(probsTh)
    cumProbsTh = np.insert(cumProbsTh,0,0)
    # ----- Simulation -----
    # NumPy array with values given a probability
    vals = np.random.choice([0,1], p=[1-probs[p],probs[p]], size=(exp,trials))
    # Calculate sum of successes
    sums = np.sum(vals, axis=1)
    # Binomial simulated stats
    meanSim = np.mean(sums)
    varSim = np.std(sums)**2.0
    skSim = skew(sums,bias=False)
    means.append(meanSim)
    variances.append(varSim)
    skewnesses.append(skSim)
    # ----- Comparison -----
    print(f" Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
    print(f" Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
    print(f" Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
    # ----- Plot -----
    if plot:
      # PMF
      axes[0].hist(sums, bins=points, density=True, color=f"C{p}", histtype="step", label=f"Simulation (p={probs[p]},n={trials})")
      axes[0].plot(probsTh, marker=".", linestyle="None", color=f"C{p}", label=f"Theory (p={probs[p]},n={trials})")
      # CMF
      axes[1].hist(sums, bins=points, density=True, cumulative=True, color=f"C{p}", histtype="step", label=f"Simulation (p={probs[p]},n={trials})")
      axes[1].plot(cumProbsTh, marker=".", linestyle="None", color=f"C{p}", label=f"Theory (p={probs[p]},n={trials})")
  # Save plot
  if plot:
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    plt.savefig("binomialDists.pdf")
  # Return stats
  return means, variances, skewnesses

def hypergeometricDist(N: int, K: int, n: int, trials: int, plot: bool)->Tuple[float,float,float]:
  '''
  Hypergeometric distribution of a random variable
  Arguments:
    N       : population size (i.e., 100)
    K       : number of success states in population (i.e., 50)
    n       : number of draws in each trial
    trials  : number of trials
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # ----- Theory -----
  # Expected value
  meanTh = n*K/N
  # Variance
  varTh = n*(K/N)*((N-K)/N)*((N-n)/(N-1))
  # Skewness
  skNum = (N-2*K)*(N-1)**0.5*(N-2*n)
  skDenom = (n*K*(N-K)*(N-n))**0.5*(N-2)
  skTh = skNum/skDenom
  # Probabilities
  #  Finding k successes is given by
  #   f(k) = [(K choose k) (N-K choose n-k)] / (N choose n)
  points = np.arange(0,K+2,1)
  probsTh = [math.comb(K,k)*math.comb(N-K,n-k)/math.comb(N,n) for k in points[:-1]]
  cumProbsTh = np.cumsum(probsTh)
  cumProbsTh = np.insert(cumProbsTh,0,0)
  # ----- Simulation -----
  # Population vector
  Nvec = np.concatenate((np.zeros(N-K),np.ones(K)), axis=0)
  np.random.shuffle(Nvec)
  idx = np.random.rand(trials,N).argpartition(n,axis=1)[:,:n]
  nvec = np.take(Nvec, idx)
  # Sums
  sums = np.sum(nvec,axis=1)
  # Stats
  meanSim = np.mean(sums)
  varSim = np.std(sums)**2
  skSim = skew(sums,bias=False)
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
  # ----- Plot -----
  if plot:
    # PMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,K)
    plt.title(f"Hypergeometric Distribution PMF (N={N}, K={K}, n={n})")
    plt.hist(sums, bins=points, density=True, histtype="step", label=f"Simulation")
    plt.plot(probsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("X")
    plt.ylabel("P(X)")
    plt.legend(frameon=False)
    plt.savefig("hypergeometricDistPMF.pdf")
    # CMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,K)
    plt.title(f"Hypergeometric Distribution CDF (N={N}, K={K}, n={n})")
    plt.hist(sums, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation")
    plt.plot(cumProbsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("X")
    plt.ylabel("Cumulative Sum")
    plt.legend(frameon=False)
    plt.savefig("hypergeometricDistCDF.pdf")
  return meanSim, varSim, skSim

def hypergeometricDistRawSim(N: int, K: int, n: int, trials: int)->np.ndarray:
  '''
  Hypergeometric distribution of a random variable
  Arguments:
    N       : population size (i.e., 100)
    K       : number of success states in population (i.e., 50)
    n       : number of draws in each trial
    trials  : number of trials
  Returns:
    sums    : numpy array with the sum of successes in each experiment
  '''
  # ----- Simulation -----
  # Population vector
  Nvec = np.concatenate((np.zeros(N-K),np.ones(K)), axis=0)
  np.random.shuffle(Nvec)
  idx = np.random.rand(trials,N).argpartition(n,axis=1)[:,:n]
  nvec = np.take(Nvec, idx)
  # Sums
  sums = np.sum(nvec,axis=1)
  return sums

def hypergeometricDists(N: int, Ks: list, ns: list, trials: int, plot: bool)->Tuple[list,list,list]:
  '''
  Hypergeometric distribution of a random variable
  Arguments:
    N       : population size (i.e., 100)
    Ks      : list of number of success states in population (i.e., [50,60,70])
    ns      : list of number of draws in each trial (i.e., [100,200,300])
    trials  : number of trials
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # Start plot
  if plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    axes[0].set_title("Hypergeometric Distributions PMF")
    axes[0].set_xlim(0,max(Ks))
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("P(k)")
    axes[1].set_title("Hypergeometric Distributions CDF")
    axes[1].set_xlim(0,max(Ks))
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Cumulative Sum")
  # Loop through various draw scenarios
  means = []
  variances = []
  skewnesses = []
  scenarios = zip(Ks,ns)
  colorCounter = 0
  for K, n in scenarios:
    colorCounter += 1
    print(f"Scenario: N={N}, K={K}, n={n}")
    # ----- Theory -----
    # Expected value
    meanTh = n*K/N
    # Variance
    varTh = n*(K/N)*((N-K)/N)*((N-n)/(N-1))
    # Skewness
    skNum = (N-2*K)*(N-1)**0.5*(N-2*n)
    skDenom = (n*K*(N-K)*(N-n))**0.5*(N-2)
    skTh = skNum/skDenom
    # Probabilities
    #  Finding k successes is given by
    #   f(k) = [(K choose k) (N-K choose n-k)] / (N choose n)
    points = np.arange(0,K+2,1)
    probsTh = [math.comb(K,k)*math.comb(N-K,n-k)/math.comb(N,n) for k in points[:-1]]
    cumProbsTh = np.cumsum(probsTh)
    cumProbsTh = np.insert(cumProbsTh,0,0)
    # ----- Simulation -----
    # Population vector
    Nvec = np.concatenate((np.zeros(N-K),np.ones(K)), axis=0)
    np.random.shuffle(Nvec)
    idx = np.random.rand(trials,N).argpartition(n,axis=1)[:,:n]
    nvec = np.take(Nvec, idx)
    # Sums
    sums = np.sum(nvec,axis=1)
    # Stats
    meanSim = np.mean(sums)
    varSim = np.std(sums)**2
    skSim = skew(sums,bias=False)
    means.append(meanSim)
    variances.append(varSim)
    skewnesses.append(skSim)
    # ----- Comparison -----
    print(f" Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
    print(f" Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
    print(f" Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
    # ----- Plot -----
    if plot:
      # PMF
      axes[0].hist(sums, bins=points, density=True, histtype="step", color=f"C{colorCounter}", label=f"Simulation (N={N}, K={K}, n={n})")
      axes[0].plot(probsTh, marker=".", linestyle="None", color=f"C{colorCounter}", label=f"Theory (N={N}, K={K}, n={n})")
      # CMF
      axes[1].hist(sums, bins=points, density=True, cumulative=True, histtype="step", color=f"C{colorCounter}", label=f"Simulation (N={N}, K={K}, n={n})")
      axes[1].plot(cumProbsTh, marker=".", linestyle="None", color=f"C{colorCounter}", label=f"Theory (N={N}, K={K}, n={n})")
  # End plot
  if plot:
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    plt.savefig("hypergeometricDists.pdf")
  # Return stats
  return means, variances, skewnesses

def geometricDist(p: float, limit: int, trials: int, plot: bool)->Tuple[float,float,float]:
  '''
  Geometric distribution of a random variable.
  Arguments:
    p       : Success probability
    limit   : Max number of trials to consider
    trials  : Number of experiments to simulate (i.e. 10k)
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # ----- Theory -----
  # Expected value
  meanTh = 1/p
  # Variance
  varTh = (1-p)/p**2
  # Skewness
  skTh = (2-p)/np.sqrt(1-p)
  # Probabilities
  #  Finding first success after k trials
  #   f(k) = p*(1-p)^(k-1)
  points = np.arange(1,limit+2,1)
  probsTh = [p*((1-p)**(k-1)) for k in points[:-1]]
  cumProbsTh = np.cumsum(probsTh)
  probsTh = np.insert(probsTh,0,0)
  cumProbsTh = np.insert(cumProbsTh,0,0)
  # ----- Simulation -----
  # Samples
  samples = np.random.geometric(p=p, size=trials)
  # Stats
  meanSim = np.mean(samples)
  varSim = np.std(samples)**2
  skSim = skew(samples,bias=False)
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
  # ----- Plot -----
  if plot:
    # PMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,limit)
    plt.title(f"Geometric Distribution PMF (p={p})")
    plt.hist(samples, bins=points, density=True, histtype="step", label=f"Simulation")
    plt.plot(probsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("k")
    plt.ylabel("P(X=k)")
    plt.legend(frameon=False)
    plt.savefig("geometricDistPMF.pdf")
    # CMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,limit)
    plt.title(f"Geometric Distribution CDF (p={p})")
    plt.hist(samples, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation")
    plt.plot(cumProbsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("k")
    plt.ylabel("Cumulative Sum")
    plt.legend(frameon=False)
    plt.savefig("geometricDistCDF.pdf")
  return meanSim, varSim, skSim

def geometricDistRawSim(p: int, trials: int)->np.ndarray:
  '''
  Geometric distribution of a random variable
  Arguments:
    p       : Success probability
    limit   : Max number of trials to consider
    trials  : number of trials
  Returns:
    samples : numpy array with the number of trials to get the first success
  '''
  # ----- Simulation -----
  # Samples
  samples = np.random.geometric(p=p, size=trials)
  return samples

def geometricDists(ps: list, limit: int, trials: int, plot: bool)->Tuple[float,float,float]:
  '''
  Geometric distribution of a random variable.
  Arguments:
    ps      : List of success probabilities
    limit   : Max number of trials to consider
    trials  : Number of experiments to simulate (i.e. 10k)
    plot    : option to plot
  Returns:
    meanSim : simulated means
    varSim  : simulated variances
    skSim   : simulated skewnesses
  '''
  # Start plot
  if plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    axes[0].set_title("Geometric Distributions PMF")
    axes[0].set_xlim(0, limit)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("P(k)")
    axes[1].set_title("Geometric Distributions CDF")
    axes[1].set_xlim(0,limit)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Cumulative Sum")
  # Loop through various draw scenarios
  means = []
  variances = []
  skewnesses = []
  colorCounter = 0
  for p in ps:
    colorCounter += 1
    print(f"Scenario: p={p}")
    # ----- Theory -----
    # Expected value
    meanTh = 1/p
    # Variance
    varTh = (1-p)/p**2
    # Skewness
    skTh = (2-p)/np.sqrt(1-p)
    # Probabilities
    #  Finding first success after k trials
    #   f(k) = p*(1-p)^(k-1)
    points = np.arange(1,limit+2,1)
    probsTh = [p*((1-p)**(k-1)) for k in points[:-1]]
    cumProbsTh = np.cumsum(probsTh)
    probsTh = np.insert(probsTh,0,0)
    cumProbsTh = np.insert(cumProbsTh,0,0)
    # ----- Simulation -----
    # Samples
    samples = np.random.geometric(p=p, size=trials)
    # Stats
    meanSim = np.mean(samples)
    varSim = np.std(samples)**2
    skSim = skew(samples,bias=False)
    means.append(meanSim)
    variances.append(varSim)
    skewnesses.append(skSim)
    # ----- Comparison -----
    print(f" Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
    print(f" Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
    print(f" Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
    # ----- Plot -----
    if plot:
      # PMF
      axes[0].hist(samples, bins=points, density=True, histtype="step", color=f"C{colorCounter}", label=f"Simulation (p={p})")
      axes[0].plot(probsTh, marker=".", linestyle="None", color=f"C{colorCounter}", label=f"Theory (p={p})")
      # CMF
      axes[1].hist(samples, bins=points, density=True, cumulative=True, histtype="step", color=f"C{colorCounter}", label=f"Simulation (p={p})")
      axes[1].plot(cumProbsTh, marker=".", linestyle="None", color=f"C{colorCounter}", label=f"Theory (p={p})")
  # End plot
  if plot:
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    plt.savefig("geometricDists.pdf")
  # Return stats
  return means, variances, skewnesses

def uniformDist(a:float, b:float, trials: int, plot:bool)->Tuple[float,float,float]:
  '''
  Uniform distribution of a random variable.
  Arguments:
    a       : lower limit (i.e., 0)
    b       : upper limit (i.e., 1)
    trials  : Number of trials to simulate (i.e. 10k)
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # ----- Theory -----
  # Expected value
  meanTh = 0.5*(b+a)
  # Variance
  varTh = (1/12)*(b-a)**2
  # Skewness
  skTh = 0
  # Probabilities
  #  Success probability of drawing x
  #  f(x) = 1/(b-a)
  width = 1e-1
  points = np.arange(a,b+width,width)
  probTh = 1/(b-a)
  cumProbsTh = [(p-a)*probTh for p in points[:-1]]
  # ----- Simulation -----
  # Samples
  samples = np.random.uniform(low=a, high=b, size=trials)
  # Stats
  meanSim = np.mean(samples)
  varSim = np.std(samples)**2
  skSim = skew(samples,bias=False)
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
  # ----- Plot -----
  if plot:
    # PMF
    plt.figure(figsize=(8,6))
    plt.xlim(0,b+a)
    plt.title(f"Uniform Distribution PMF (a={a},b={b})")
    plt.hist(samples, bins=points, density=True, histtype="step", label=f"Simulation")
    plt.axhline(y=probTh, xmin=a/(b+a), xmax=b/(b+a), color="grey", linestyle="-", label=f"Theory")
    plt.xlabel("x")
    plt.ylabel("P(X=x)")
    plt.legend(frameon=False)
    plt.savefig("uniformDistPMF.pdf")
    # CMF
    plt.figure(figsize=(8,6))
    plt.xlim(a,b)
    plt.title(f"Uniform Distribution CDF (a={a},b={b})")
    plt.hist(samples, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation")
    plt.plot(points[:-1], cumProbsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("x")
    plt.ylabel("Cumulative Sum")
    plt.legend(frameon=False)
    plt.savefig("uniformDistCDF.pdf")
  return meanSim, varSim, skSim

def uniformDistRawSim(a: float, b: float, trials: int)->np.ndarray:
  '''
  Uniform distribution of a random variable.
  Arguments:
    a       : lower limit (i.e., 0)
    b       : upper limit (i.e., 1)
    trials  : Number of trials to simulate (i.e. 10k)
  Returns:
    samples : numpy array with all the trials from sampling the uniform distribution
  '''
  # ----- Simulation -----
  # Samples
  samples = np.random.uniform(low=a, high=b, size=trials)
  return samples

def uniformDists(ais: float, bes: float, trials: int, plot: bool)->Tuple[list,list,list]:
  '''
  Uniform distribution of a random variable.
  Arguments:
    ais     : lower limits (i.e., 0)
    bes     : upper limits (i.e., 1)
    trials  : Number of trials to simulate (i.e. 10k)
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # Start plot
  if plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    axes[0].set_title("Uniform Distributions PMF")
    axes[0].set_xlim(min(ais), max(bes))
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("P(k)")
    axes[1].set_title("Uniform Distributions CDF")
    axes[1].set_xlim(min(ais),max(bes))
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Cumulative Sum")
  # Loop through various draw scenarios
  means = []
  variances = []
  skewnesses = []
  scenarios = zip(ais,bes)
  colorCounter = 0
  for a,b in scenarios:
    colorCounter += 1
    print(f"Scenario: a={a}, b={b}")
    # ----- Theory -----
    # Expected value
    meanTh = 0.5*(b+a)
    # Variance
    varTh = (1/12)*(b-a)**2
    # Skewness
    skTh = 0
    # Probabilities
    #  Success probability of drawing x
    #  f(x) = 1/(b-a)
    width = 1e-1
    points = np.arange(a,b+width,width)
    probTh = 1/(b-a)
    cumProbsTh = [(p-a)*probTh for p in points[:-1]]
    # ----- Simulation -----
    # Samples
    samples = np.random.uniform(low=a, high=b, size=trials)
    # Stats
    meanSim = np.mean(samples)
    varSim = np.std(samples)**2
    skSim = skew(samples,bias=False)
    means.append(meanSim)
    variances.append(varSim)
    skewnesses.append(skSim)
    # ----- Comparison -----
    print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
    print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
    print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
    # ----- Plot -----
    if plot:
      # PMF
      axes[0].hist(samples, bins=points, density=True, histtype="step", label=f"Simulation (a={a},b={b})")
      axes[0].axhline(y=probTh, xmin=a/(b+a), xmax=b/(b+a), color="grey", linestyle="-", label=f"Theory (a={a},b={b})")
      # CMF
      axes[1].hist(samples, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation (a={a},b={b})")
      axes[1].plot(points[:-1], cumProbsTh, marker=".", linestyle="None", label=f"Theory (a={a},b={b})")
  # End plot
  if plot:
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    plt.savefig("uniformDists.pdf")
  # Return stats
  return means, variances, skewnesses

def normalDist(mu: float, sigma: float, trials: int, plot: bool)->Tuple[float,float,float]:
  '''
  Normal distribution of a random variable
  Arguments:
    mu      : mean of distribution
    sigma   : variance of distribution
    trials  : number of simulations (i.e., 10k)
    plot    : option to plot
  Returns:
    meanSim : simulated mean
    varSim  : simulated variance
    skSim   : simulated skewness
  '''
  # ----- Theory -----
  # Expected value
  meanTh = mu
  # Variance
  varTh = sigma**2
  # Skewness
  skTh = 0
  # Probabilities
  #  Success probability of drawing x
  #  f(x) = 1/sqrt(2*pi*sigma^2) e^(-(x-mu)^2/(2*sigma^2))
  width = sigma/5.
  points = np.arange(mu-5*sigma,mu+5*sigma+width,width)
  probsTh = [1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mu)**2)/(2*sigma**2)) for x in points[:-1]]
  cumProbsTh = np.cumsum(probsTh)*width
  # ----- Simulation -----
  # Samples
  samples = np.random.normal(loc=mu, scale=sigma, size=trials)
  # Stats
  meanSim = np.mean(samples)
  varSim = np.std(samples)**2
  skSim = skew(samples,bias=False)
  # ----- Comparison -----
  print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
  print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
  print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
  # ----- Plot -----
  if plot:
    # PMF
    plt.figure(figsize=(8,6))
    plt.xlim(mu-5*sigma,mu+5*sigma)
    plt.title(f"Normal Distribution PMF (mu={mu},std={sigma})")
    plt.hist(samples, bins=points, density=True, histtype="step", label=f"Simulation")
    plt.plot(points[:-1], probsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("x")
    plt.ylabel("P(X=x)")
    plt.legend(frameon=False)
    plt.savefig("normalDistPMF.pdf")
    # CMF
    plt.figure(figsize=(8,6))
    plt.xlim(mu-5*sigma,mu+5*sigma)
    plt.title(f"Normal Distribution CDF (mu={mu},std={sigma})")
    plt.hist(samples, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation")
    plt.plot(points[:-1], cumProbsTh, marker=".", linestyle="None", label=f"Theory")
    plt.xlabel("x")
    plt.ylabel("Cumulative Sum")
    plt.legend(frameon=False)
    plt.savefig("normalDistCDF.pdf")
  return meanSim, varSim, skSim

def normalDistRawSim(mu: float, sigma: float, trials: int, plot: bool)->np.ndarray:
  '''
  Normal distribution of a random variable
  Arguments:
    mu      : mean of distribution
    sigma   : variance of distribution
    trials  : number of simulations (i.e., 10k)
    plot    : option to plot
  Returns:
    samples : numpy array with all the trials from sampling the normal distribution
  '''
  # ----- Simulation -----
  # Samples
  samples = np.random.normal(loc=mu, scale=sigma, size=trials)
  return samples

def normalDists(mus: list, sigmas: list, trials: int, plot: bool)->Tuple[list,list,list]:
  '''
  Normal distribution of a random variable
  Arguments:
    mus     : list of means of distributions
    sigmas  : list of stds of distributions
    trials  : number of simulations (i.e., 10k)
    plot    : option to plot
  Returns:
    samples : numpy array with all the trials from sampling the normal distribution
  '''
  # Start plot
  if plot:
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
    axes[0].set_title("Normal Distributions PMF")
    axes[0].set_xlim(min(mus)-5*max(sigmas), min(mus)+5*max(sigmas))
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("P(k)")
    axes[1].set_title("Normal Distributions CDF")
    axes[1].set_xlim(min(mus)-5*max(sigmas), min(mus)+5*max(sigmas))
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Cumulative Sum")
  # Loop through various draw scenarios
  means = []
  variances = []
  skewnesses = []
  scenarios = zip(mus,sigmas)
  colorCounter = 0
  for mu,sigma in scenarios:
    colorCounter += 1
    print(f"Scenario: mu={mu}, std={sigma}")
    # ----- Theory -----
    # Expected value
    meanTh = mu
    # Variance
    varTh = sigma**2
    # Skewness
    skTh = 0
    # Probabilities
    #  Success probability of drawing x
    #  f(x) = 1/sqrt(2*pi*sigma^2) e^(-(x-mu)^2/(2*sigma^2))
    width = sigma/5.
    points = np.arange(mu-5*sigma,mu+5*sigma+width,width)
    probsTh = [1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mu)**2)/(2*sigma**2)) for x in points[:-1]]
    cumProbsTh = np.cumsum(probsTh)*width
    # ----- Simulation -----
    # Samples
    samples = np.random.normal(loc=mu, scale=sigma, size=trials)
    # Stats
    meanSim = np.mean(samples)
    varSim = np.std(samples)**2
    skSim = skew(samples,bias=False)
    means.append(meanSim)
    variances.append(varSim)
    skewnesses.append(skSim)
    # ----- Comparison -----
    print(f"Simulated mean is {meanSim:.5f} (expected {meanTh:.5f})")
    print(f"Simulated variance is {varSim:.5f} (expected {varTh:.5f})")
    print(f"Simulated skewness is {skSim:.5f} (expected {skTh:.5f})")
    # ----- Plot -----
    if plot:
      # PMF
      axes[0].hist(samples, bins=points, density=True, histtype="step", label=f"Simulation (mu={mu},std={sigma})")
      axes[0].plot(points[:-1], probsTh, marker=".", linestyle="None", label=f"Theory (mu={mu},std={sigma})")
      # CMF
      axes[1].hist(samples, bins=points, density=True, cumulative=True, histtype="step", label=f"Simulation (mu={mu},std={sigma})")
      axes[1].plot(points[:-1], cumProbsTh, marker=".", linestyle="None", label=f"Theory (mu={mu},std={sigma})")
  # End plot
  if plot:
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    plt.savefig("normalDists.pdf")
  # Return stats
  return means, variances, skewnesses

if __name__ == "__main__":
  normalDists([2,3,4],[0.25,0.5,0.75],100000,True)
