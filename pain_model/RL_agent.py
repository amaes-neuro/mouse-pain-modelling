import torch
import torch.optim as optim
import random
from RL_model import QNetwork

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

device = torch.device("cuda" if use_cuda else "cpu")
from  torch.autograd import Variable

from replay_buffer import ReplayMemory, Transition


class Agent(object):

    def __init__(self, n_states, n_actions, hidden_dim, name, capacity, batch_size, learning_rate):
        """Agent class that choose action and train
        Args:
            n_states (int): input dimension
            n_actions (int): output dimension
            hidden_dim (int): hidden dimension
        """
        
        self.replace_target_cnt = 12e4
        self.learn_step_counter = 0
        self.q_local = QNetwork(n_states, n_actions, hidden_dim=16, name=name+'_q_local').to(device)
        self.q_target = QNetwork(n_states, n_actions, hidden_dim=16, name=name+'_q_target').to(device)
        self.q_target.load_state_dict(self.q_local.state_dict())
        self.q_target.eval()
        
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.q_local.parameters(), lr=learning_rate)
        
        self.n_states = n_states
        self.n_actions = n_actions

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        #  ReplayMemory: trajectory is saved here
        self.replay_memory = ReplayMemory(capacity)
        

    def get_action(self, state, eps, check_eps=True):
        """Returns an action
        Args:
            state : 2-D tensor of shape (n, input_dim)
            eps (float): eps-greedy for exploration
        Returns: int: action index
        """
        sample = random.random()

        if check_eps==False or sample > eps:
            self.q_local.eval()
            with torch.no_grad():
               # t.max(1) will return largest column value of each row.
               # second column on max result is index of where max element was
               # found, so we pick action with the larger expected reward.
               action = self.q_local(Variable(state).type(FloatTensor)).data.max(1)[1].view(1, 1)
            self.q_local.train()
            return action
        else:
            ## return LongTensor([[random.randrange(2)]])
            return torch.tensor([[random.randrange(self.n_actions)]], device=device) 


    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_target.load_state_dict(self.q_local.state_dict())
            self.q_target.eval()
            
    def save_models(self):
        self.q_local.save_checkpoint()
        self.q_target.save_checkpoint()


    def load_models(self):
        self.q_local.load_checkpoint()
        self.q_target.load_checkpoint()           


    def learn(self, experiences=None, gamma=0.999):
        """Prepare minibatch and train them
        Args:
        experiences (List[Transition]): batch of `Transition`
        gamma (float): Discount rate of Q_target
        """
        
        if len(self.replay_memory.memory) < self.batch_size:
            return;
         
        if experiences is None:
            experiences = self.replay_memory.sample(self.batch_size)
            
        self.optimizer.zero_grad() 
        
        self.replace_target_network()
        
        batch = Transition(*zip(*experiences))
                        
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        next_states = torch.cat(batch.next_state)
        dones = torch.cat(batch.done)
        
            
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to newtork q_local (current estimate)
        Q_expected = self.q_local(states).gather(1, actions)     

        with torch.no_grad():
            Q_targets_next = self.q_target(next_states).detach().max(1)[0] 
            #Q_targets_next = self.q_local(next_states).detach().max(1)[0]
            Q_targets = rewards + (gamma * Q_targets_next * (1-dones))
        
        #self.q_local.train(mode=True)        
        
        loss = self.mse_loss(Q_expected, Q_targets.unsqueeze(1))
        loss.backward()
        for param in self.q_local.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.learn_step_counter += 1

