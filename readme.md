# blackjack solver

The goal of this project is to create a blackjack solver. For a given player hand and the dealer's upcard, it should figure out whether the best action is to hit or stand.

There are currently two notebooks in the project:

- `monte_carlo.ipynb` estimates the probability of the dealer ending on each possible total. It starts with a Monte Carlo simulation and then uses dynamic programming to calculate the same probabilities.
- `q_learning.ipynb` uses Q-learning to learn the best action for different blackjack states. It also keeps track of how often each state and action has been visited, which helps reduce the variance from random exploration.

For now, the solver only supports hitting and standing. Aces are handled as either 1 or 11, depending on whether counting them as 11 would make the hand go over 21.
## next steps

- add doubling down
- add splitting
- add surrendering
- compare the learned policy with the exact probabilities
- explore more when hitting and standing have similar values
