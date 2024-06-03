# MDP-and-RL-for-Grid-World
This project implements Grid World Value Iteration and Q-Learning algorithms. It includes scripts for running value iteration (ValueIteration.py) and Q-learning (QLearning.py), which print utility values, optimal policies, and display relevant plots. Data files for various grid configurations are provided.

Grid World Value Iteration and Qlearning

Requirements:
- Python 3.x
- NumPy
- Matplotlib
# pip install numpy matplotlib

1. Value Iteration
	use ValueIteration.py main sorces code for value iteration
		utility values and policy will be printed and the 
		two plots of grid wold and the utility convergence will be desplayed 
		
	I have modified the give data file for the value iteration (4x3 and 4x4 worlds) to make the compatable while reading the data file
	The data files are save as 4x3.txt and 4x4.txt
	another addtional 4x4_2.txt is prepared to cahnge the reward values in the state

	Run the script from the terminal by providing the path to the data file as an argument.
	sample bash is provided below for each data file

		1. python3 ValueIteration.py 4x3.txt
		2. python3 ValueIteration.py 4x3_2.txt
		3. python3 ValueIteration.py 4x4.txt
		4. python3 ValueIteration.py 4x4_2.txt
		5. python3 ValueIteration.py 4x4_gamma.txt

2. Q-Learning
	use QLearning.py (arguments --iteratio, --epsilon, --datafile)
	thus the iteration epsilon can be cahnged
		Optimal Policy, utility values (maximun Q value) and Q values will be printed
		and the grid word with the q value list in each sate is displayed
	6x6 world is prepared for testing the q learnig with 6x6.txt data file
	
	sample bash is provided below for each data fill 
	(default values epsilon 0.05, iteration 10,000 and datafile 4x4.txt)
	
	
		1. python3 QLearning.py --epsilon 0.2 --iteration 10000 --data_file 4x4.txt
		2. python3 QLearning.py --epsilon 0.2 --iteration 10000 --data_file 6x6.txt


