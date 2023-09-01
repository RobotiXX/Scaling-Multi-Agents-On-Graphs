import numpy as np
import networkx as nx
import itertools
import random
import matplotlib.pyplot as plt
import utils
import argparse
import time

import gymnasium as gym
from gymnasium import spaces
from graph import Graph
import torch

# stable_baselines3 imports
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_checker import check_env

# Maskable PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete, InvalidActionEnvMultiBinary, InvalidActionEnvMultiDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker



class MultiCombinationGraph:
    def __init__(self, mim_distance_to_goal, num_nodes, risk_edge_ratio=0.5):
        self.total_graphs_with_risk = []
        self.total_graphs_without_risk = []
        self.mim_distance_to_goal = mim_distance_to_goal
        self.num_nodes = num_nodes
        self.risk_edge_ratio = risk_edge_ratio

    # you have only one graph     
    def add_list_of_connnected_graph_without_risk(self):
        # Create a new list to store graphs without risky edges
        # Filter out graphs that are not connected
        connected_graphs_without_risk = []
        
        # Generate all possible edges
        all_possible_edges = [(i, j) for i in range(self.num_nodes) for j in range(i + 1, self.num_nodes)]

        # Generate power set of all possible edges (all combinations including empty set)
        power_set_of_edges = list(itertools.chain.from_iterable(
            itertools.combinations(all_possible_edges, r) for r in range(len(all_possible_edges) + 1)))
        for edges in power_set_of_edges:
            G = nx.Graph()
            G.add_nodes_from(range(self.num_nodes))
            G.add_edges_from(edges)
            start_node = 0
            goal_node = self.num_nodes - 1

            if nx.is_connected(G):
                # Check that the distance between start_node and goal_node is at least min_distance
                distance = nx.shortest_path_length(G, source=start_node, target=goal_node)
                if distance >= self.mim_distance_to_goal:
                    connected_graphs_without_risk.append(G)

        if connected_graphs_without_risk != []:
            self.total_graphs_without_risk = connected_graphs_without_risk
            return self.total_graphs_without_risk
        raise ValueError("No connected graphs without risk edges found")
    
    def add_list_of_connnected_graph_with_risk(self):

        # Create a new list to store graphs with risky edges
        connected_graphs_with_risk = []

        if self.total_graphs_without_risk !=[]:
            self.total_graphs_without_risk = self.add_list_of_connnected_graph_without_risk()

        # Add "risky" edges to each connected graph
        for G_old in self.total_graphs_without_risk:
            edges = list(G_old.edges())
            num_risky_edges = len(edges) // 2  # 1/2 of total edges to be risky
            all_combinations_of_risky_edges = list(itertools.combinations(edges, num_risky_edges))
            if len(all_combinations_of_risky_edges) > 0:
            
                for risky_edges in all_combinations_of_risky_edges:

                    G_new = G_old.copy()
                    # Add unique edges from G_new.edges() to unique_edges
                    G_new.graph['unique_edges'] = sorted(list(G_new.edges()))
                    # Add risky edges as a graph attribute
                    G_new.graph['risk_edges'] = risky_edges  
                    # Add risky edges with support nodes as another graph attribute
                    G_new.graph['risk_edges_with_support_nodes'] = {edge: edge for edge in risky_edges}
                    connected_graphs_with_risk.append(G_new)


                if connected_graphs_with_risk != []:
                    self.total_graphs_with_risk = connected_graphs_with_risk
                    print("self.total_graphs_with_risk", self.total_graphs_with_risk)
                    return self.total_graphs_with_risk
                raise ValueError("No connected graphs with risk edges found")
                
    
    def save_list_of_connnected_graph_with_risk(self):
        # Save graph to a pickle file
        utils.save_to_pickle_file(self.total_graphs_with_risk, '5_nodes_graph.pkl')

    def load_list_of_connnected_graph_with_risk(self):
        # To read graph back from pickle file
        return utils.load_from_pickle_file('5_nodes_graph.pkl')

class MultiCombinationGraphWorldEnv(gym.Env):
    def __init__(self, n_agents, nodes, edges, risky_edges_with_support, render_mode="test"): 
        super(MultiCombinationGraphWorldEnv, self).__init__()
        self.render_mode = render_mode

        ## one hot encoded information
        self.agent_position_one_hot = None
        self.goal_position_one_hot = None

        ## non one hot encoded information
        self.agent_position = None
        self.goal_position = None

        self.observation = None
        self.num_agents = n_agents
        self.num_nodes = len(nodes)
        self.nodes = nodes
        self.edges = edges
        self.risky_edges_with_support_nodes = risky_edges_with_support
        ## env information
        self.graph = Graph(nodes, edges)
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
        print("Its invalid action! Stay at same place!")
        return position, valid_move
    
    def single_agent_step(self, action, agent_id):
        #1. Store the previous position for rollback in case of a wall
        #2. If the new position is a wall, revert to the previous position
        #3. Add a penalty if the agent didn't move

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
        
        new_position = action.copy().tolist()
        done = False
        truncated = False

        ## only if both the agents are able to move then only it will be valid move
        ## reward shaping
        step_cost = 0
        wall_penalty = 0
        coordination_reward = 0
        goal_reward = 0
        distance_to_goal_reward = 0


        ## distance to goal reward, TO DO
        epsilon = 100
        total_distance_to_goal = 0
        for agent_id in range(self.num_agents):
                shortest_path_length = self.graph.shortest_path_length(
                    new_position[agent_id], self.goal_position[agent_id]
                )
                total_distance_to_goal += shortest_path_length

        avg_distance_to_goal = total_distance_to_goal / self.num_agents
        distance_to_goal_reward = 1 / (avg_distance_to_goal + epsilon)

        print("avg_distace_to_goal", avg_distance_to_goal)
        print("distance_to_goal_reward", distance_to_goal_reward)

        ## wall/stagnant penalty plus no coordination
        if np.array_equal(new_position, prev_position):
            reward = -1 + distance_to_goal_reward                          # wall_penalty + step_cost
            # reward = -1               
            self.agent_position = prev_position
            new_agent_obs = self.observation.copy()
        else:
            
            
            # time.sleep(5)
            if new_position == self.goal_position:
                done = True
                reward = +10 # 100, -1, -10 or 10, -0.01, -1

            ## step cost
            else:
                reward = -0.01 + distance_to_goal_reward                   # step_cost + distance_to_goal_reward
                # reward = -0.01

            # reward = step_cost + wall_penalty + coordination_reward + distance_to_goal_reward + goal_reward
            new_agent_obs = self.create_new_observation()
            # time.sleep(100)
            self.agent_position = new_position
            for agent_id in range(self.num_agents):
                new_agent_obs[agent_id][action[agent_id]] = 1
    
        self.observation = new_agent_obs
        
    
        print("------------------------------")
        print("prev_position: ", prev_position)
        print("action: ", action)
        print("reward: ", reward)
        print("new position: ", new_position)
        print("new obs:", new_agent_obs)
        print("------------------------------")
        new_obs = np.array(new_agent_obs).flatten()
        if done == True:
            print("done: ", done)
        print("-------------New Step Done-----------------")
        return new_obs, reward, done, truncated , {}
    
    def valid_action_mask(self):
        print("--------inside generate_masks--------")
        print(f"states {self.observation} && type {type(self.observation)}")
       
        # Determine valid actions for each agent
        valid_actions = []
        for i in range(self.num_agents):
            valid_actions_agent = list(self.graph.get_neighbors(self.agent_position[i]))
            # valid_actions_agent.append(non_zero_indices[i][0])
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

# save and load the traing graph data
def load_graph_data(args):
    print("------------------Load Graph Data Started--------------------------")
    GS = MultiCombinationGraph(args.mim_distance_to_goal, args.num_of_nodes, args.risk_edge_ratio)
    ### save combination of graphs with risky edges to a pickle file
    GS.add_list_of_connnected_graph_without_risk()  # <---- Call this first
    GS.add_list_of_connnected_graph_with_risk() # <---- Call this second
    GS.save_list_of_connnected_graph_with_risk() # <---- Call this third
    ### load combination of graphs with risky edges from a pickle file
    loaded_G_data = GS.load_list_of_connnected_graph_with_risk() # <---- Call this fourth
    

    for i in range(len(loaded_G_data)):
        print("Graph ", i)
        print("Nodes",loaded_G_data[i].nodes())
        print("Edges",loaded_G_data[i].graph['unique_edges'])
        print("Risk Edges",loaded_G_data[i].graph['risk_edges'])
        print("Risk Edges with support nodes",loaded_G_data[i].graph['risk_edges_with_support_nodes'])

    print("Number of connected graphs total: ", len(GS.total_graphs_without_risk))
    print("Number of connected graphs with risk edges: ", len(loaded_G_data))
    print("------------------Load Graph Data Done--------------------------")
    return loaded_G_data
# Register the environment in the gym as "MultiCombinationGraph-v0"
gym.register(
    id='MultiCombinationGraph-v0',
    entry_point=MultiCombinationGraphWorldEnv,
)

def make_env(rank, seed=42):
    def _init():
        env = MultiCombinationGraphWorldEnv()  # Assuming CustomGridWorld is your environment class
        env = Monitor(env, filename="multicombination_graphworld_monitor", allow_early_resets=True)
        env.reset(seed=seed + rank)
        # check_env(env, warn=True) # check if env is running
        return env
    return _init

def mask_fn(env: gym.Env) -> np.ndarray :
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

def make_env(graph, mask_fn):
    env = gym.make('MultiCombinationGraph-v0',
                   n_agents=2,
                   nodes=graph.nodes(),
                   edges=graph.graph['unique_edges'],
                   risky_edges_with_support=graph.graph['risk_edges_with_support_nodes'])
    env = ActionMasker(env, mask_fn)  # If you're using action masking
    return env
from functools import partial
def train_multiple_env(train_graphs, tensorboard_log):
    # train_graphs = load_graph_data(args)

    # Shuffle the training graphs
    random.shuffle(train_graphs)

    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    batch_size = 6  
    # Initialize model outside of loop
    first_graph = train_graphs[0]
    print("first_graph", first_graph)
    # first_env = gym.make('MultiCombinationGraph-v0',
    #                     n_agents=2,
    #                     nodes=first_graph.nodes(),
    #                     edges=first_graph.graph['unique_edges'],
    #                     risky_edges_with_support=first_graph.graph['risk_edges_with_support_nodes'])
    # first_env = ActionMasker(first_env, mask_fn)
    env = partial(make_env, graph=first_graph, mask_fn=mask_fn)
    first_env = DummyVecEnv([env]*batch_size)
    model = MaskablePPO(MaskableActorCriticPolicy,
                        env=first_env,
                        device=device,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        n_steps=1000,
                        tensorboard_log=tensorboard_log)
    
    # Batching   
    if len(train_graphs) >= batch_size:
        for i in range(0, len(train_graphs), batch_size):
            batch_graphs = train_graphs[i:i + batch_size]
            if len(batch_graphs) < batch_size:
                break
            # envs = [gym.make('MultiCombinationGraph-v0',
            #                 n_agents=2,
            #                 nodes=graph.nodes(),
            #                 edges=graph.graph['unique_edges'],
            #                 risky_edges_with_support=graph.graph['risk_edges_with_support_nodes']) for graph in batch_graphs]
        
            # envs = [ActionMasker(env, mask_fn) for env in envs]  # If you're using action masking
            env_fns = [partial(make_env, graph=graph, mask_fn=mask_fn) for graph in batch_graphs]

            vec_env = DummyVecEnv(env_fns)
            # Convert it to a vectorized environment
            # vec_env = DummyVecEnv(envs)
            model.set_env(vec_env)
            model.learn(total_timesteps=50000, use_masking=True)

    # Save the final model
    model.save("model/ppo_parallel_envs_multicombination_graphworld")
    del model

def train_sequential_env(train_graphs, tensorboard_log):
    print("------------------Train Started--------------------------")
    # Shuffle the training graphs
    random.shuffle(train_graphs)
    train_graphs = train_graphs[:2]

   

    # Initialize model outside of loop
    first_graph = train_graphs[0]
    print("first_graph", first_graph)
    nodes = first_graph.nodes()
    edges = first_graph.graph['unique_edges']
    risk_edges_with_support_nodes = first_graph.graph['risk_edges_with_support_nodes']
    first_env = gym.make('MultiCombinationGraph-v0',
                        n_agents=2,
                        nodes=nodes,
                        edges=edges, 
                        risky_edges_with_support=risk_edges_with_support_nodes)
    first_env = ActionMasker(first_env, mask_fn)
    policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])
    model = MaskablePPO(MaskableActorCriticPolicy,
                        env=first_env,
                        device=device,
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        n_steps=100, 
                        tensorboard_log=tensorboard_log)

    for graph in train_graphs:
        nodes = graph.nodes()
        edges = graph.graph['unique_edges']
        risk_edges_with_support_nodes = graph.graph['risk_edges_with_support_nodes']
        env = gym.make('MultiCombinationGraph-v0',
                        n_agents=2,
                        nodes=nodes, 
                        edges=edges, 
                        risky_edges_with_support=risk_edges_with_support_nodes)
        env = ActionMasker(env, mask_fn)  # If you're using action masking

        model.set_env(env)
        model.learn(total_timesteps=10000, use_masking=True)

    # Save the final model
    model.save("model/ppo_sequential_envs_multicombination_graphworld")
    del model # remove to demonstrate saving and loading
    print("------------------Train Done--------------------------")

def evaluate_env(test_graphs, train_type="sequential"):
    print("------------------Test Started--------------------------")
    # Shuffle the training graphs
    random.shuffle(test_graphs)

    # Initialize model outside of loop
    test_graph = test_graphs[0] # you can do random choice here
    nodes = test_graph.nodes()
    edges = test_graph.graph['unique_edges']
    risk_edges_with_support_nodes = test_graph.graph['risk_edges_with_support_nodes']
    test_env = gym.make('MultiCombinationGraph-v0',
                        n_agents=2,
                        nodes=nodes, 
                        edges=edges, 
                        risky_edges_with_support=risk_edges_with_support_nodes)
    test_env = ActionMasker(test_env, mask_fn)

    if train_type == "parallel":
        model = MaskablePPO.load("model/ppo_parallel_envs_multicombination_graphworld", env=test_env)
    elif train_type == "sequential":
        model = MaskablePPO.load("model/ppo_sequential_envs_multicombination_graphworld", env=test_env)

    vec_env = model.get_env() # train_vec_env
    obs = vec_env.reset()
    print("obs: ", obs)
    total_steps = 0
    total_path = []
    total_path.append(vec_env.envs[0].agent_position)
    for i in range(100):
        total_steps += 1
        action, _states = model.predict(obs, action_masks=mask_fn(test_env)) # action_masks=mask_fn(env
        obs, rewards, dones, info = vec_env.step(action)
        print("info: ", info)
        total_path.append(action)
        time.sleep(1)
        if any(dones) == True:
            break
    print("total_steps: ", total_steps)
    print("total_path: ", total_path)
    print("------------------Test Done--------------------------")

def arg_parse():
    parser = argparse.ArgumentParser(description="Grid Style Graph World")
    # Argument for setting random seed
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for initialization")
    # Argument for the number of agents
    parser.add_argument('--num_of_agents', type=int, default=2,
                        help="Number of agents in the environment")
    # Argument for risk_edge_ratio
    parser.add_argument('--risk_edge_ratio', type=float, default=0.5,
                        help="The ratio of risky edges in the graph")
    # Argument for the number of nodes
    parser.add_argument('--num_of_nodes', type=int, default=5,
                        help="Number of nodes in the graph")
    parser.add_argument('--mim_distance_to_goal', type=int, default=2,
                        help="Minimum distance between start and goal nodes")
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    '''
    Note: Needs to Support for 2 agents only
    Steps:
    1. Add/Load a 5 nodes graph without risk.
    2. Calculate different combinations of the same graph without risk.
    3. Add risk to each combination of the graph and save it to a pickle file.
    4. Load the pickle file and use it for training using PPO.
    5. Chose a random graph from the pickle file and use it for training using PPO or 
       Chose a multiple random graph from the pickle file and use it for training using PPO.
    6. Repeat steps 4 and 5 for different number of agents.

    5 steps in elaboration: 
    '''
    args = arg_parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


    ## data loading
    train_G_data = load_graph_data(args)
    print("train_G_data", train_G_data)
    ## data sampling
    # sample one graph from the training data
    sampled_G = random.choice(train_G_data)
    nodes = sampled_G.nodes()
    edges = sampled_G.graph['unique_edges']
    risk_edges = sampled_G.graph['risk_edges']
    risk_edges_with_support_nodes = sampled_G.graph['risk_edges_with_support_nodes']
    print("sampled_G", sampled_G)
    print("edges", edges)
    print("risk_edges_with_support_nodes", risk_edges_with_support_nodes)
    ## Enable TensorBoard logging
    tensorboard_log = "./tensorboard_log/"+str(args.num_of_agents)+"_agents_"+str(args.num_of_nodes)+"_nodes/"

    ## Sequential training
    ## train the model
    train_sequential_env(train_G_data, tensorboard_log) # works
    # evaluate the model
    evaluate_env(train_G_data, train_type="sequential") # works


    ## Parallel training
    # train_multiple_env(train_G_data) # works
    # evaluate_env(train_G_data, train_type="parallel") # works but suboptimal

    
   



    
























'''
Two approaches for training:

1) Multiple graphs with risk edges
- train agents across multiple graphs
- This approach is akin to the idea of domain randomization, 
where training across multiple environments can lead to a more robust policy

2) One graph with risk edges

3) combining both approaches
- sometimes sampling random graphs and sometimes sticking with a single graph for multiple trajectories.
- It's okay for different graphs to yield trajectories of different lengths. Just make sure you're properly 
handling these in your training batches (for example, by padding and masking if necessary).

- Depending on the size of your graphs and the length of the trajectories, you may have to think carefully 
about how to store these trajectories. If you find you're running out of memory, you may need to write the 
trajectories to disk or employ some form of experience replay mechanism where older trajectories are discarded.

'''

