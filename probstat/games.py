import numpy as np

"""

  Pass Line Bet Craps (Vegas Rules):

    The game involves rolling two six-sided dice.
    First ("come-out") roll:
      a. If the outcome is a 7 or 11, the pass line bet wins (called a "natural")
      b. If the outcome is 2, 3 or 12, the pass line bet losses (called "craps")
      c. If the outcome is 4, 5, 6, 8, 9 or 10, that number becomes the point
    Point phase:
      Once the point is established, the shooter continues rolling
      a. If the shooter rolls the point again, pass line bets win
      b. If the shooter rolls a 7 before the point, pass line bets lose  (called "seven out")

"""

def craps()->bool:
  '''
  Pass Line Bet Craps (Vegas Rules)
  Simulation of an individual game.
  Returns:
    bool: True if game won, false if game lost
  '''
  # Come-out roll
  comeOut = np.sum(np.random.randint(low=1, high=7, size=2))
  # Evaluate come-out roll
  # Natural
  if comeOut in [7,11]:
    return True
  # Crap
  elif comeOut in [2,3,12]:
    return False
  # Point phase
  else:
    while True:
      point = np.sum(np.random.randint(low=1,high=7,size=2))
      if point == comeOut:
        return True
      elif point == 7:
        return False

def multipleCraps(games: int)->float:
  '''
  Pass Line Bet Craps (Vegas Rules)
  Simulation of multiple games.
  Arguments:
    games: Integer specifying the number of simulations to run
  Returns:
    float: Winning rate
  '''
  # Simulate games
  crapsSim = [craps() for _ in range(games)]
  # Calculate winning rate
  winRate = np.mean(crapsSim)
  # Expected winning rate
  #  P(win) = P(come-out win) + P(win | point 4)*P(point 4) + P(win | point 5)*P(point5) + ...
  #  Note that P(win | point x) = n/(n+6), where n is the number of ways x can occur. 
  #   From the whole space, n outputs result in a win and 6 outputs result in 7, which is a losing condition.
  #  P(win) = 8/36 + [3/(3+6)]*3/36 + [4/(4+6)]*4/36 + ... = 244/495
  winRateExp = 244/495
  print(f"Percentge of wins in {games} games of craps: {winRate*100:.2f}% (expected: {winRateExp*100:.2f}%)")
  # Return winning rate
  return winRate
  
  

if __name__=="__main__":
  multipleCraps(100000)
