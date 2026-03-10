import numpy as np

"""
  oneDiceRoll: 
    Simulate the roll of a six-sided fair dice.

  twoDiceRollSum:
    Simulate the sum of rolling two six-sided fair dice.
"""

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
