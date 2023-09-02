import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import networkx as nx
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import random
import torch
import os

# from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_checker import check_env

### Maskable PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete, InvalidActionEnvMultiBinary, InvalidActionEnvMultiDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker

# from mask_ActorCrticPolicy import MaskableActorCriticPolicy
# from ppo_mask import MaskablePPO

## Two Agents GraphProblem
class Graph:
    def __init__(self, nodes: Dict[int, Tuple[int]], edges: Dict[Tuple[int, int], float]):
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        self.start_goal = 0
        self.end_goal = len(self.graph.nodes) - 1
    
    def get_neighbors(self, node):
        return list(self.graph.neighbors(node))
    
    def get_fixed_start_goal(self):
        return self.start_goal
    
    def get_fixed_end_goal(self):
        return self.end_goal

    def get_random_start_goal(self):
        nodes = list(self.graph.nodes)
        random_index = random.randint(0, len(nodes) - 1)
        random_node = nodes[random_index]
        self.start_goal = random_node
        return self.start_goal
    
    def get_random_end_goal(self):
        nodes = list(self.graph.nodes)
        while True:
            random_index = random.randint(0, len(nodes) - 1)
            random_node = nodes[random_index]

            if random_node != self.start_goal:
                break
        self.end_goal = random_node
        return self.end_goal
    
    def get_random_node_from_list(self, nodes):
        random_index = random.randint(0, len(nodes) - 1)
        random_node = nodes[random_index]
        return random_node   

    def shortest_path(self, start, goal):
        return nx.shortest_path(self.graph, start, goal)
    
    def shortest_path_length(self, start, goal):
        return nx.shortest_path_length(self.graph, start, goal)

class CustomGridWorld(gym.Env):
    def __init__(self, num_agents, render_mode="test"): 
        super(CustomGridWorld, self).__init__()
        self.render_mode = render_mode
        ### agent_position 
        self.agent_position_one_hot = None
        ### goal information
        self.goal_position_one_hot = None

        self.observation = None

        self.agent_position = None
        self.goal_position = None
        self.num_agents = num_agents

        ## graph information
        ## 5 nodes 
        ## 10000 time steps, 100 steps works fine : with mask (before test)
        self.nodes = {0:(0), 1:(1), 2:(2), 3:(3), 4:(4)}

        # unordered edges
        self.edges = {(0, 2):1, (1, 3):1, (1, 4):1, (2, 3):1} ## sparse graph 
        self.risk_edges_with_support_nodes = {(0, 2): (0, 2), (1, 3): (1, 3)}

        # ordered edges
        # final testing being done
        # self.edges = {(0, 1):1, (1, 2):1, (2, 3):1, (3, 4):1} ## sparse graph 
        # self.risk_edges_with_support_nodes = {(0, 1):(0, 1), (2, 3): (2, 3)}  
        # path taken : [array([1, 0]), array([1, 1]), array([2, 2]), array([3, 2]), array([3, 3]), array([4, 4])]
        # time taken : 59 seconds, 108 secs
        # 100 steps, 10000 time steps, 200 steps, 10000 time steps,  200 steps, 20000 steps all works fine
        # model : [128]*3, surprised by this


        # self.edges = {(0, 1):1, (0, 2):1, (1, 2):1, (2, 3):1, (3, 4):1} # moderate graph
        # self.risk_edges_with_support_nodes = {(0, 2): (0, 2), (2, 3): (2, 3)} # adjacent risk edges
        # ## path taken : [array([2, 0]), array([2, 2]), array([3, 2]), array([3, 3]), array([4, 4])]
        # ## time taken : 102.86 secs
        # model : [128]*3, surprised by this
        ## 20100 time steps, 300 steps works fine : with mask

        # self.edges = {(0, 1):1, (0, 2):1,(1, 2):1,(1, 3):1, (2, 3):1,(3, 4):1} # dense graph
        # self.risk_edges_with_support_nodes = {(0, 1): (0, 1),(2, 3): (2, 3)} 
        # this will result in issue

        # self.edges = {(0, 3):1, (1, 2):1, (3, 1):1, (1, 4):1, (3, 2):1, (2, 4):1} # 300 steps
        # self.risk_edges_with_support_nodes = {(0, 3): (0, 3), (3, 1): (3, 1), (1, 4): (1, 4), (3, 2): (3, 2)} 
        ## path: [array([0, 3]), array([3, 3]), array([3, 2]), array([2, 2]), array([4, 4])]
        # model : [128]*3, surprised by this
        ## 20100 time steps, 300 steps works fine : with mask
        # time taken: 102. 07 secs



        ## 10 nodes     
        # self.nodes = {0:(0), 1:(1), 2:(2), 3:(3), 4:(4), 5:(5), 6:(6), 7:(7), 8:(8), 9:(9)}
        # self.edges = {(0, 1):1, (1, 2):1, (2, 3):1, (3, 4):1, (4, 5):1, (5, 6):1, (6, 7):1, (7, 8):1, (8, 9):1} ## sparse graph
        # self.risk_edges_with_support_nodes = {(0, 1):(0, 1), (2, 3): (2, 3), (4, 5): (4, 5), (6, 7):(6, 7)}
        


        # self.edges = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1,(1, 4): 1,(1, 5): 1, (2, 3): 1,(2, 4): 1, 
        #             (2, 6): 1, (3, 6): 1, (3, 5): 1, (3, 4): 1, (4, 5): 1, (4, 6): 1, 
        #             (5, 7): 1, (5, 6): 1, (6, 7): 1, (6, 8): 1, (7, 8): 1, (7, 9): 1,
        #             (8, 4): 1, (8, 9): 1}
        #risk_edges_with_support_nodes = {(0, 3):(0, 3), (5, 7): (5, 7)}
        
        
        # ## 15 nodes
        # self.nodes = {0: (0), 1: (1), 2: (2), 3: (3), 4: (4), 5: (5), 6: (6), 7: (7),
        #               8: (8), 9: (9), 10: (10), 11: (11), 12: (12), 13: (13), 14: (14)}
        # self.edges = {(0, 1): 1, (0, 2): 1, (1, 2): 1,(1, 6): 1, (2, 3): 1,(2, 4): 1, 
        #         (2, 4): 1, (3, 6): 1, (3, 8): 1,  (4, 5): 1, (4, 8): 1, (4, 9): 1, 
        #         (5, 7): 1, (5, 8): 1, (6, 7): 1, (6, 8): 1,  (7, 8): 1, (7, 9): 1,
        #          (8, 4): 1, (8, 10):1,  (8, 12):1, (8, 13):1, (8, 14):1, 
        #          (9, 10):1, (9, 13):1, (9, 14):1, (10, 12):1, (10, 14):1, (11, 12):1,
        #          (11, 13):1, (12, 13):1,(12, 14):1}
        # self.edges = {(0, 1): 1, (0, 2): 1, (0, 4): 1, (1, 2): 1,(1, 3): 1,(1, 6): 1,
        #      (2, 3): 1,(2, 4): 1, (2, 9): 1, (2, 13): 1,
        #     (2, 9): 1, (3, 6): 1, (3, 8): 1, (3, 4): 1, (3, 11): 1,
        #     (4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1, (4, 11): 1,
        #     (5, 7): 1, (5, 8): 1, (5, 9): 1, (6, 7): 1, (6, 8): 1, (6, 9): 1, (6, 12): 1,
        #     (7, 8): 1, (7, 9): 1,(7, 11): 1, (7, 13): 1, (2, 7): 1,
        #      (9, 8): 1, (8, 10):1, (8, 11):1, (8, 12):1,(8, 13):1, (8, 14):1, 
        #      (9, 10):1, (9, 11):1,(9, 13):1, (9, 13):1, (10, 12):1, (10, 13):1, (10, 14):1, (11, 12):1,
        #      (11, 13):1, (12, 13):1,(12, 14):1}
        # # ## 50000 time steps, 500 steps works fine : with mask standard
        # self.edges = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
        #                 (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1}
        # self.risk_edges_with_support_nodes = {(0, 1): (0, 1), (3, 4): (3, 4),  (5, 6): (5, 6), (9, 10): (9, 10), (12, 13): (12, 13)}
        
        

        
        # ## 20 nodes
        # self.nodes = {0: (0), 1: (1), 2: (2), 3: (3), 4: (4), 5: (5), 6: (6), 7: (7), 8: (8), 9: (9), 10: (10), 11: (11), 12: (12), 13: (13), 14: (14),
        #      15: (15), 16: (16), 17: (17), 18: (18), 19: (19)}
        # self.edges = {(0, 1): 1, (0, 2): 1, (0, 4): 1, (1, 2): 1,(1, 3): 1,(1, 6): 1, (2, 3): 1,(2, 4): 1, 
        #         (2, 10): 1, (3, 6): 1, (3, 8): 1, (3, 4): 1, (4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1, 
        #         (5, 7): 1, (5, 8): 1, (5, 9): 1, (6, 7): 1, (6, 8): 1, (6, 9): 1, (7, 8): 1, (7, 9): 1,(7, 10): 1,(7, 11): 1,(7, 13): 1,
        #         (8, 7): 1, (8, 9): 1, (8, 10):1, (8, 11):1, (8, 12):1,(8, 13):1, (8, 14):1, (8, 15):1, (8, 16):1,(8, 17):1,(8, 18):1,(8, 19):1,
        #         (9, 10):1, (9, 11):1, (9, 13):1, (9, 14):1, (10, 12):1, (10, 14):1, (11, 12):1, (11, 14):1,(11, 15):1, (11, 16):1, (11, 19):1,
        #         (11, 13):1, (12, 13):1,(12, 14):1, (12, 16):1, (12, 18):1, (13, 14):1, (13, 15):1, (13, 18):1,
        #         (14, 15):1, (14, 16):1, (14, 17):1, (14, 18):1, (14, 19):1, (15, 16):1, (15, 17):1, (15, 18):1,
        #         (16, 17):1,  (17, 18):1, (17, 19):1, (18, 19):1}
        # self.edges = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
        #                 (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1, (14, 15): 1, (15, 16): 1,
        #                 (16, 17): 1, (17, 18): 1, (18, 19): 1}
        # self.risk_edges_with_support_nodes = {(0, 1): (0, 1), (3, 4): (3, 4),  (6, 7): (6, 7), (9, 10): (9, 10), (12, 13): (12, 13),
        #                                         (16, 17): (16, 17)}

        
        # ## 25 nodes

        # self.nodes = {0: (0), 1: (1), 2: (2), 3: (3), 4: (4), 5: (5), 6: (6), 7: (7), 8: (8), 9: (9), 10: (10), 11: (11), 12: (12), 13: (13), 14: (14),
        #      15: (15), 16: (16), 17: (17), 18: (18), 19: (19), 20: (20), 21: (21), 22: (22), 23: (23), 24: (24)}
        # self.edges = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1,(1, 3): 1,(1, 6): 1, (2, 3): 1,(2, 4): 1, 
        #         (2, 8): 1, (3, 6): 1, (3, 8): 1, (3, 4): 1, (4, 5): 1, (4, 6): 1, (4, 7): 1, (4, 9): 1, (4, 11): 1, (4, 21): 1, 
        #         (5, 7): 1, (5, 8): 1, (5, 9): 1, (6, 7): 1, (6, 8): 1, (6, 9): 1, (7, 8): 1, (7, 9): 1,(7, 10): 1,(7, 11): 1,(7, 13): 1,(7, 24): 1,
        #         (8, 4): 1, (8, 9): 1, (8, 10):1, (8, 11):1, (8, 12):1,(8, 13):1, (8, 14):1, (8, 15):1, (8, 16):1,(8, 17):1,(8, 18):1,(8, 19):1, (8, 19): 1,
        #         (9, 10):1, (9, 11):1, (9, 13):1, (9, 14):1, (10, 12):1, (10, 14):1, (11, 12):1, (11, 14):1,(11, 15):1, (11, 16):1, (11, 19):1,(11, 24):1,
        #         (11, 13):1, (12, 13):1,(12, 14):1, (12, 16):1, (12, 18):1, (13, 14):1, (13, 15):1, (13, 18):1, (13, 21):1, (8, 23):1,
        #         (14, 15):1, (14, 16):1, (14, 17):1, (14, 18):1, (14, 19):1, (15, 16):1, (15, 17):1, (15, 18):1, (13, 22):1,
        #         (16, 17):1,  (17, 18):1, (17, 19):1, (18, 19):1, (18, 19):1, (18, 20):1, (18, 19):1, (18, 20):1, (19, 20):1, (19, 22):1,
        #         (18, 21):1, (18, 22):1, (18, 23):1, (18, 24):1, (20, 21):1, (19, 23):1, (19, 24):1, (20, 22):1, (20, 23):1, (20, 24):1}

        # self.edges = {(0, 1): 1, (1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1,
        #                 (8, 9): 1, (9, 10): 1, (10, 11): 1, (11, 12): 1, (12, 13): 1, (13, 14): 1, (14, 15): 1, (15, 16): 1,
        #                 (16, 17): 1, (17, 18): 1, (18, 19): 1, (19, 20):1, (20, 21):1, (21, 22):1, (22, 23):1, (23, 24):1 }
        # self.risk_edges_with_support_nodes = {(0, 1): (0, 1), (3, 4): (3, 4),  (6, 7): (6, 7), (9, 10): (9, 10), (12, 13): (12, 13),
        #                                         (15, 16): (15, 16), (18, 19): (18, 19), (21, 22): (21, 22)}


        # ## 30 nodes
        # self.nodes = {0: (0), 1: (1), 2: (2), 3: (3), 4: (4), 5: (5), 6: (6), 7: (7), 8: (8), 9: (9), 10: (10), 11: (11), 12: (12), 
        #               13: (13), 14: (14), 15: (15), 16: (16), 17: (17), 18: (18), 19: (19), 20: (20), 21: (21), 22: (22), 23: (23),
        #                 24: (24), 25: (25), 26: (26), 27: (27), 28: (28), 29: (29)}
        
        # self.edges = {(0, 1): 1, (0, 2): 1, (0, 3): 1, (1, 2): 1,(1, 3): 1,(1, 6): 1, 
        #      (2, 3): 1,(2, 4): 1, (2, 8): 1, (3, 6): 1, (3, 8): 1, (3, 4): 1,
        #      (4, 7): 1, (4, 9): 1, (4, 11): 1, (5, 7): 1, (5, 8): 1, (5, 9): 1, 
        #      (6, 7): 1, (6, 8): 1, (6, 9): 1, (7, 10): 1,(7, 13): 1,(7, 24): 1,
        #      (8, 9): 1, (8, 10):1, (8, 11):1, (8, 16):1,(8, 17):1, (8, 19): 1,
        #      (9, 10):1, (9, 11):1, (9, 13):1, (9, 14):1, (9, 23):1,
        #      (10, 12):1, (10, 14):1, (10, 23):1, (10, 25):1, (10, 29):1,
        #      (11, 14):1,(11, 15):1, (11, 16):1, (11, 19):1,(11, 24):1,
        #      (12, 13):1,(12, 14):1, (12, 16):1, (12, 18):1, 
        #      (13, 14):1, (13, 15):1, (13, 18):1, (13, 21):1, (13, 29):1, 
        #      (14, 15):1, (14, 16):1, (14, 17):1, (14, 18):1, (14, 19):1, 
        #      (15, 16):1, (15, 17):1, (15, 18):1, (13, 22):1,
        #      (16, 17):1,  (17, 18):1, (17, 19):1, (17, 27):1, 
        #      (18, 19):1,  (18, 20):1, (18, 23):1, (18, 24):1,
        #      (19, 20):1, (19, 22):1, (20, 21):1, (19, 23):1, (19, 24):1,
        #      (20, 22):1, (20, 23):1, (20, 24):1, (20, 26):1,
        #      (21, 23):1, (21, 28):1, (27, 28):1, (22, 29):1,
        #      (22, 26):1, (23, 27):1, (24, 28):1,(25, 29):1,}
 


        ## env information
        self.graph = Graph(self.nodes, self.edges)
        self.action_space = spaces.MultiDiscrete([len(self.nodes)]*self.num_agents)  # 5 actions per agent
        # The first two dimensions are for the 'position' and the next 16 are for the 4x4 'grid'
        # self.observation_space = spaces.MultiDiscrete([4, 4] + [2]*16)  # Assuming grid can have 2 types of cells, 0 and 1
        self.observation_space = spaces.MultiDiscrete([2]*len(self.nodes)*(self.num_agents+1))  # Assuming grid can have 2 types of cells, 0 and 1
        self.reset()

    def create_new_observation(self, reset=False):
        # Create the dynamic observation array
        self.observation = np.zeros((self.num_agents + 1, len(self.nodes)), dtype=int)
        if reset == True:
             for agent in range(self.num_agents):
                self.observation[agent, 0] = 1
        else: # for updating the observation
            self.observation[:, 0] = 0
        self.observation[self.num_agents, -1] = 1
        return self.observation

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.observation = self.create_new_observation(reset=True)
        ### agent_position,goal_position 
        self.agent_position_one_hot = self.observation[:self.num_agents,:]
        self.goal_position_one_hot = self.observation[self.num_agents] # last row of observation
        self.agent_position = [0]*self.num_agents
        self.goal_position = [len(self.nodes)-1]*self.num_agents

        # print("agent_position: ", self.agent_position_one_hot)
        # print("goal_position: ", self.goal_position_one_hot)
        # print("reset return: ", self.observation)

        # time.sleep(1000)
        return self.observation.flatten(), {}
    

    
    def check_valid_move(self, position, action):
        valid_move = False
        if action in self.graph.get_neighbors(position):
            print("Its a valid action!Agent moves to new position!")
            valid_move = True
            return action, valid_move
        print("Its valid action! Agent stays at same place!")
        return position, valid_move
    
    def single_agent_step(self, action, agent_id):
        #1. Store the previous position for rollback in case of a wall
        #2. If the new position is a wall, revert to the previous position
        #3.Add a penalty if the agent didn't move

        one_agent_prev_position = self.agent_position[agent_id]
        one_agent_new_position = self.agent_position[agent_id]

        ## take new action for each agent to get new position
        one_agent_new_position, valid_move = self.check_valid_move(one_agent_prev_position, action)
        return one_agent_new_position, one_agent_prev_position, valid_move
        
    
    def step(self, action): 
        print("-------------New Step Started-----------------")
        # Store the previous position for to add penalty if agent didnt move
        prev_position = self.agent_position.copy()
       
        ## this doesnot affect due to current action masking
        action1 = action[0]
        action2 = action[1]
        new_position1, _, validmove1 = self.single_agent_step(action1, 0)
        new_position2, _ , validmove2 = self.single_agent_step(action2, 1)

        new_position = [new_position1, new_position2]
        # print(type(action))
        
        # new_position = action.copy().tolist()
        done = False
        truncated = False

        ## only if both the agents are able to move then only it will be valid move
        ## reward shaping
        step_cost = 0
        wall_penalty = 0
        coordination_reward = 0
        non_coordination_penalty = 0 # to do
        goal_reward = 0
        path_length_penalty = 0 # to do
        distance_to_goal_reward = 0

    
        if not np.array_equal(new_position, prev_position):
            edge1 = (prev_position[0], new_position[0])
            edge2 = (prev_position[1], new_position[1])   
            print("edge1: ", edge1)
            print("edge2: ", edge2)
            # both moves, when both moves no support mechanism is used
            if prev_position[0] != new_position[0] and prev_position[1] != new_position[1]:
                print("both moves")
                ## both moves in risky
                if edge1 in self.risk_edges_with_support_nodes.keys() \
                    and edge2 in self.risk_edges_with_support_nodes.keys():
                    coordination_reward = -5
                ## if one moves in risky and other in safe
                elif edge1 in self.risk_edges_with_support_nodes.keys() \
                    or edge2 in self.risk_edges_with_support_nodes.keys():
                    coordination_reward = -5
                ## both moves in safe
                else:
                    coordination_reward = +1
            ## 1st stay, 2nd move 
            elif prev_position[0] == new_position[0] and prev_position[1] != new_position[1]:
                print("1st stay, 2nd move")
                if edge2 in self.risk_edges_with_support_nodes.keys():
                    support_nodes = self.risk_edges_with_support_nodes[edge2]
                    print("support_nodes: ", support_nodes)
                    ## (no support mechanism is used)
                    if prev_position[0] not in support_nodes:
                        coordination_reward = -5 # -3, +1 also works
                    ## (support mechanism is used)
                    elif prev_position[0] in support_nodes:
                        coordination_reward = +2
            ## 2nd stay, 1st move
            elif prev_position[0] != new_position[0] and prev_position[1] == new_position[1]:
                print("2nd stay, 1st move")
                if edge1 in self.risk_edges_with_support_nodes.keys():
                    support_nodes = self.risk_edges_with_support_nodes[edge1]
                    print("support_nodes: ", support_nodes)
                    ## (no support mechanism is used)
                    if prev_position[1] not in support_nodes:
                        coordination_reward = -5
                    ## (support mechanism is used)
                    elif prev_position[1] in support_nodes:
                        coordination_reward = +2


        ## distance to goal reward, TO DO
        epsilon = - 0.05 #(closer to goal, higher reward, less negative value)
        total_distance_to_goal = 0
        for agent_id in range(self.num_agents):
                shortest_path_length = self.graph.shortest_path_length(
                    new_position[agent_id], self.goal_position[agent_id]
                )
                total_distance_to_goal += shortest_path_length

        avg_distance_to_goal = total_distance_to_goal / self.num_agents
        distance_to_goal_reward =  epsilon * avg_distance_to_goal

        print("avg_distace_to_goal", avg_distance_to_goal)
        print("distance_to_goal_reward", distance_to_goal_reward)
        print("coordination_reward", coordination_reward)
        
        # closer to goal, higher reward, more positive value
        coordination_factor = 1.2**(-avg_distance_to_goal)
        print("coordination_factor", coordination_factor)
        # time.sleep(1)

        ## wall/stagnant penalty plus no coordination
        if np.array_equal(new_position, prev_position):
            # reward = -5 + distance_to_goal_reward                          # wall_penalty + step_cost
            # reward = -1
            wall_penalty = -5               
            self.agent_position = prev_position
            new_agent_obs = self.observation.copy()
        else:
            
            
            
            # time.sleep(5)
            if new_position == self.goal_position:
                done = True
                # reward = +10 + distance_to_goal_reward                 # 100, -1, -10 or 10, -0.01, -1
                goal_reward = +10
            ## step cost
            else:
                # reward = -0.01 + distance_to_goal_reward    # step_cost + distance_to_goal_reward, stable
                # reward = -0.01 + distance_to_goal_reward + coordination_reward*(-distance_to_goal_reward)  # most stable works 
                # # reward = -0.01 + distance_to_goal_reward + coordination_reward*(2+distance_to_goal_reward) # works 
                # reward = -0.01 + distance_to_goal_reward + coordination_reward*(coordination_factor)  #suboptimal              
                # reward = -0.01
                step_cost = -0.01
            
            
            
            
            
            new_agent_obs = self.create_new_observation()
            # time.sleep(100)
            self.agent_position = new_position
            self.agent_position = new_position
            for agent_id in range(self.num_agents):
                new_agent_obs[agent_id][action[agent_id]] = 1
    
        reward = step_cost + wall_penalty + coordination_reward*(-distance_to_goal_reward) + distance_to_goal_reward + goal_reward
        self.observation = new_agent_obs
        
    
        print("------------------------------")
        print("prev_position: ", prev_position)
        print("action: ", action)
        print("reward: ", reward)
        print("new position: ", new_position)
        print("new obs:", new_agent_obs)
        print("------------------------------")
        new_obs = np.array(new_agent_obs).flatten()
        # time.sleep(100)
        # time.sleep(1)
        # time.sleep(5)
        if done == True:
            print("done: ", done)
            # new_obs = self.reset()
            # time.sleep(5)
        print("-------------New Step Done-----------------")
        return new_obs, reward, done, truncated , {}
    
    def valid_action_mask(self):
        print("--------inside generate_masks--------")
        print(f"states {self.observation} && type {type(self.observation)}")
       
        # Determine valid actions for each agent
        valid_actions = []
        for i in range(self.num_agents):
            valid_actions_agent = list(self.graph.get_neighbors(self.agent_position[i]))
            valid_actions_agent.append(self.agent_position[i])
            valid_actions.append(valid_actions_agent)
        # Create masks for each agent
        total_actions = len(self.nodes)
        masks = [[[1 if j in valid_actions_agent else 0 for j in range(total_actions)]] for valid_actions_agent in valid_actions] # Ensure mask tensors are on the same device as states
        print("masks: ", masks)
        ## masks:  [tensor([[0, 1, 0, 0, 0]]), tensor([[0, 1, 0, 0, 0]])] valid 1, invalid 0
        return np.array(masks)
    
    def render(self):
        pass
    def close(self):
        return super().close()
    
# Wrap your environment with the Monitor wrapper
# Register the custom environment
gym.register(
    id='CustomGridWorld-v0',
    entry_point=CustomGridWorld,
)

def make_env(rank, seed=42):
    def _init():
        env = CustomGridWorld()  # Assuming CustomGridWorld is your environment class
        env = Monitor(env, filename="gridworld_monitor", allow_early_resets=True)
        env.reset(seed=seed + rank)
        # check_env(env, warn=True)
        return env
    return _init

def mask_fn(env: gym.Env) -> np.ndarray :
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    start_time = time.time()
    # n_envs = 1
    # envs = [make_env(i) for i in range(n_envs)]
    # train_vec_env = SubprocVecEnv(envs) # better parallelization than DummyVecEnv
    # eval_vec_env = SubprocVecEnv(envs)
    num_agents = 2
    num_nodes = 10
    env_id = "CustomGridWorld-v0"
    env = gym.make(env_id, num_agents=2)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    # env.valid_action_mask()
    # check_env(env, warn=True)

    # time.sleep(1000)
    # Enable TensorBoard logging

    tensorboard_log = "./tensorboard_log/"+str(num_agents)+"_agents_"+str(num_nodes)+"_nodes/"
    policy_kwargs = dict(net_arch=[dict(pi=[128]*3, vf=[128]*3)]) ### works very well 5 nodes
    # policy_kwargs = dict(net_arch=[dict(pi=[256]*2, vf=[256]*2)]) #### works very well for 10 nodes
    # # policy_kwargs = dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])]) # from 25 nodes
    # Set device (can be 'cuda:0', 'cuda:1', etc. in multi-GPU environments)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MaskablePPO(MaskableActorCriticPolicy,
                 env,
                 device=device,
                 policy_kwargs=policy_kwargs,
                 verbose=1,
                 tensorboard_log=tensorboard_log,
                #  gamma=0.95,
                # learning_rate=0.0001,
                 n_steps=200, # st 300
                   ) # when using MlpPolicy, the observation space must be a Box or a Discrete space

    # eval_callback = EvalCallback(eval_vec_env, best_model_save_path='./logs/',
    #                             log_path='./logs/', eval_freq= 100)
    log_dir = "."
    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    # print(model.policy)
    # print(model.policy.net_arch)
    # time.sleep(1000)
    # more comlexity, more steps, more timesteps then only it will learn
    # Train the agent
    
    model.learn(total_timesteps=20000, use_masking=True) 

    # 5 nodes, 20000 steps, 200 steps works fine : with mask standard (sparse graph)
    # 10 nodes, 50000 steps, 300 steps works fine : with mask standard (sparse graph)
    # 15 nodes, 100000 steps, 600 steps works fine : with mask standard (sparse graph) : 595.3525156974792 seconds
    # 20 nodes, 120000 steps, 800 steps works fine : with mask standard (sparse graph) : 775.9393990039825 seconds


    
    # evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=5, warn=False)

    model_name = "./ppo_graphworld"+str(num_agents)+"_agents_"+str(num_nodes)+"_nodes/"
    model.save(model_name)
    del model # remove to demonstrate saving and loading




    print("------------------Test Started--------------------------")\
    ## multiple envs
    # model = MaskablePPO.load("./ppo_graphworld", env=env)
    # vec_env = model.get_env() # train_vec_env
    # obs = vec_env.reset()
    model = MaskablePPO.load(model_name)
    vec_env = env
    obs,_  = vec_env.reset()
    print("obs: ", obs)
    total_steps = 0
    total_path = []
    total_rewards = []
    # total_path.append(vec_env.envs[0].agent_position)
    for i in range(100):
        total_steps += 1
        action, _states = model.predict(obs, action_masks=mask_fn(env)) # action_masks=mask_fn(env
        # action, _states = model.predict(obs, deterministic=True) # suboptimal
        # action, _states = model.predict(obs) # doesnot work without masking
        obs, rewards, dones, _, info = vec_env.step(action)
        # vec_env.render("human")
        # print("obs: ", obs)
        # print("rewards: ", rewards)
        # print("dones: ", dones)
        print("info: ", info)
        total_path.append(action)
        total_rewards.append(rewards)
        time.sleep(1)
        # if any(dones) == True: # multiple envs
        if dones == True:
            break
    print("------------------Test Done--------------------------")
    # print("env: ", n_envs)
    print("total_steps: ", total_steps)
    print("total_path: ", total_path)
    print("total_rewards: ", total_rewards)
    print("total_time: ", time.time() - start_time)
    print("total_timesteps: ", model.num_timesteps)
    print("total n_steps: ", model.n_steps)
    print("total n_updates: ", model.n_updates)
    # train_vec_env.close()
    # eval_vec_env.close()
    ## vew logs
## tensorboard --logdir ./tensorboard_log/ in seperate terminal



