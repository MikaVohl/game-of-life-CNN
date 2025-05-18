## Solving Conway's Game of Life using a Convolutional Neural Network

[Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is a iteration-based game with simple rules which yield suprisingly chaotic behaviour. A consequence of this chaotic behaviour is that there is no closed-form solution for finding the Nth next game state based on the current grid.

This restriction motivated me to explore using deep learning methods to attempt to predict the Nth next game state with a high level of accuracy. I quickly realized that building a convolutional neural network was the best fit for the task, as Conway's Game of Life's rules are inherently localized to a 3x3 grid, making convolutional layers with a 3x3 kernel an obvious choice.

I have deployed my implementation at [life.mikavohl.ca](life.mikavohl.ca) in the form of a blank canvas on which the user can input the initial state, then compare the results of direct simulation and the prediction by the CNN.