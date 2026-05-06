import numpy as np

from tic_tac_toe import TicTacToe


epsilon = 0.1
learning_rate = 0.15
learning_rate = 0.35

action_space = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]


def get_state_representation(state)->int:
    """ Bijective map from board state to a unique number."""
    return np.sum(3**np.arange(0, len(action_space)) * state.flatten())


class TDAgent:
    """ Temporal difference learning agent."""
    def __init__(self, value_function, player=1):
        self.player = player
        
        self.value_function = value_function
        
        self.reset_trajectory()

    # TODO: allow saving/loading state value estimates 

    # TODO: set epsilon = 0 if evaluation mode: to evaluate agent's performance
        
    def reset_trajectory(self):
        self.trajectory = []
        
    def update_trajectory(self, data):
        """ 
            Environment samples state, done, reward until episode completion.
            State value in terminal state is the sampled reward.
                1  : winning
                .5 : draw
                -1  : loose

        """
        self.trajectory.append(data)

    def update_state_values(self):
        return
    
    def choose_action(self, board):
        if np.random.rand(1) < epsilon: # epsilon greedy
            valid = False

            while not valid:
                action = np.random.randint(0,3,1), np.random.randint(0,3,1)  

                if board[action] == 0:
                    # return action
                    valid = True
        else:
            best_state_value = -1
            #action = (0,0)
            
            for a in action_space:
                if board[a] == 0:
                    board[a] = self.player # do a step

                    representation = int(get_state_representation(board)) # get state representation

                    board[a] = 0 # reset board to prev

                    state_value = self.value_function[representation]

                    if state_value >= best_state_value:
                        best_state_value = state_value
                        action = a
                    # instead using argmax?

        return action
    
    def train(self):
        """ Update value function after one episode (game). 
        """
        #for count, (state, done, reward) in enumerate(reversed(self.trajectory)):
        for (state, done, reward) in reversed(self.trajectory):
            if done:
            #if count == 0:
                v_next = reward
                continue
            
            representation = int(get_state_representation(state))
            
            self.value_function[representation] += learning_rate * (v_next - self.value_function[representation])
            
            v_next = self.value_function[representation]


def play_game(p1, p2, env, verbose=False):
    done = False
    
    while not done:    
        next_state, done, reward = env.step(p1.choose_action(env.board), player=1)
        
        p1.update_trajectory((next_state.copy(), done, reward))
        #p2.update_trajectory((next_state.copy(), done, 0))
        p2.update_trajectory((next_state.copy(), done, -reward))
    
        if verbose:
            env.render()
        
        if done:
            break
            
        #### 
        
        next_state, done, reward = env.step(p2.choose_action(env.board), player=2)
        
        p2.update_trajectory((next_state.copy(), done, reward))
        #p1.update_trajectory((next_state.copy(), done, 0))
        p1.update_trajectory((next_state.copy(), done, -reward))
        
        if verbose:
            env.render()
        
        if done:
            #print('Game finished.')
            break
        
    # episode is done
    p1.train()
    p2.train()
    
    return p1, p2, env

if __name__ == "__main__":
    env = TicTacToe()
    env.reset()
    
    p1 = TDAgent(value_function=-np.ones(env.observation_space), player=1) 
    p2 = TDAgent(value_function=-np.ones(env.observation_space), player=2) 
    
    for i in range(20000):
        if i % 2000 == 0:
            print(i)
        p1, p2, env = play_game(p1, p2, env) # agent vs agent

        # after         
        p1.reset_trajectory()
        p2.reset_trajectory()
        
        env.reset()

p1, p2, env = play_game(p1, p2, env, verbose=True) # agent vs agent
