import numpy as np

"""

  fairDiceRollEV:
    Simulate the EV of rolling a six-sided fair dice.

  uniformDrawsUntilOne:
    Single simulation of the number of draws from the Unif(0,1) to exceed 1.
  
  uniformDrawsUntilOne:
    Average number of draws from the Unif(0,1) to exceed 1.

  fairDiceRollProb:
    Simulate the probability of a specific single dice roll output.
    
"""

def fairDiceRollEV(size: int)->int:
  ''' 
  Expectation value of a fair dice roll.
  Arguments:
    size  : Simulation size (i.e., number of rolls)
  Returns:
    avg   : Average value of rolls (i.e. expectation value)
  '''
  # NumPy array with values
  vals = np.random.randint(low=1, high=7, size=size)
  # Compute the EV using the average
  avg = np.mean(vals)
  # Return probabilistic EV
  print(f"Average of {size} dice rolls is {avg}")
  return avg

def uniformDrawsUntilOne()->int:
  '''
  Single simulation of the number of draws from the uniform distribution [0,1]
   for the sum to exceed 1.
  Arguments:
    None
  Returns:
    draws   : Number of simulated draws
  '''
  draws = 0
  sum = 0
  while True:
    val = np.random.rand()
    sum += val
    draws += 1
    if sum > 1:
      return draws

def uniformDrawsUntilOneEV(size: int)->float:
  '''
  Expected value of the number of draws from the uniform [0,1] dist
   for the sum to exceed 1
  Arguments:
    size  : Numver of simulations
  Returns:
    avg   : Average number of draws
  '''
  # Simulations
  simulations = [uniformDrawsUntilOne() for _ in range(size)]
  mean = np.mean(simulations)
  # Theory EV
  #  Consider X_1 + ... + X_N > 1, where N is the number of i.i.d. draws from the Unif(0,1) needed.
  #  Then, E[N] = sum[ Pr(N>n) ], where N>n means that after n draws, the sum is still at most 1.
  #  N > n <--> X_1 + ... + X_N <= 1.
  #  E[N] = sum[ Pr(X_1 + ... + X_N <= 1) ]
  #  Note that this probability is the volume of the simplex which is 1/n!
  #  E[N] = sum[ 1/n! ] = e
  meanTh = np.exp(1)
  print(f'Simulated average number of draws is {mean} (expected {meanTh})')
  return mean

def fairDiceRollProb(successes: list[int])->float:
  '''
  Computes the probability of obtaining successes in fair six-sided dice roll.
  Arguments:
    successes   : list of positive outcomes
  Returns:
    probability : simulated probability of successes
  '''
  # Ensure successes are unique
  successes = list(set(successes))
  # Simulate 1000 dice rolls
  rolls = np.random.randint(low=1,high=7,size=1000)
  probSim = np.sum(np.isin(rolls,successes))/len(rolls)
  # Theory
  #  Six possible outcomes
  probTh = len(successes)/6
  # Return
  print(f'Simulated success probability is {probSim:.2f} (expected {probTh:.2f})')
  return probSim

#if __name__=="__main__":
