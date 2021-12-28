import numpy as np

class TicTacToe:
    def __init__(self, reward=1):
        self.size = 3
        self.reward = reward
        
        self.observation_space = 3**9
        
    def reset(self):
        self.board = np.zeros((self.size,self.size))
    
    def step(self, action, player=1):   
        """
            Interaction with environment.
            
            Return: {state, done, reward}
        """
        self.board[action] = player
        
        done, winner = self.done()

        if winner == player:
            reward = self.reward
        elif winner == 0:
            reward = 0.5
        else:
            reward = -self.reward
  
        return self.board, done, reward
    
    def done(self):
        # check diagonals
        if np.diag(self.board).tolist().count(1) == 3:
            return True, 1
        elif np.diag(self.board).tolist().count(2) == 3:
            return True, 2
        elif np.fliplr(self.board).diagonal(0).tolist().count(1) == 3:
            return True, 1
        elif np.fliplr(self.board).diagonal(0).tolist().count(2) == 3:
            return True, 2
        
        # check if draw
        elif np.all((self.board == 0) == False):
            return True, 0
        
        # check all rows and columns
        for i in range(self.size):
            if self.board[i].tolist().count(1) == 3:
                return True, 1
            elif self.board[i].tolist().count(2) == 3:
                return True, 2
            elif self.board[:,i].tolist().count(1) == 3:
                return True, 1
            elif self.board[:,i].tolist().count(2) == 3:
                return True, 2
        
        return False, 0
        
    def render(self):
        for i in range(self.size):
            print('-----------')
            for j in range(self.size):
                if self.board[i,j] == 1:
                    print(' X ', end='')
                elif self.board[i,j] == 2:
                    print(' O ', end='')
                else:
                    print('   ', end='')
                if j != 2:
                    print('|', end='')
            print()
        print('-----------')
        #print()
