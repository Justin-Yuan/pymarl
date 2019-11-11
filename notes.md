PyMARL notes 

---
## Installation 

### docker 
docker setup is for Ubuntu, automatically install StarCraft env  (with binary and maps), sacred repo and torch

to run in docker, do (where $GPU=0)
```bash 
bash run.sh 0 python3 src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s3z
```

useful Docker commands 
- docker ps 
- docker image ls/rm
- docker container ls/rm/stop/kill
- docker container exec NAME CMD 

### virtual env 
could create conda env, install dependencies from `requirements.txt` and run `install_sc2.sh`, but it only gets smac/StarCraft env for ubuntu; for mac, manually download StarCraft env from <https://starcraft2.com/en-us/>, copy the maps to `Applications/StarCraft II`

to run in virtual env, do (in activated env) 
```bash 
python src/main.py --config=qmix_smac --env-config=sc2 with env_args.map_name=2s3z
```

---
## Envs & Utilities

### StarCrafet II 
reference: <https://github.com/oxwhirl/smac>

### Sacred 
reference: <https://github.com/IDSIA/sacred>
usage:
- Experiment, SETTINGS, FileStorageObserver
- use @ex.main to tag main function 

---
## Overview 

- config is separated to default config, then load and merge other relatively static but large configs (e.g. algo & env)
- registry based: le_REGISTRY (learner), r_REGISTRY (runner), mac_REGISTRY (controller)
- `mac`: multi-agent controller
- `runner`: collect environment episode batch 
- `learner`: train policy


### Main 
- get params from `sys.argv`
- load default yaml to config_dict 
- load env config and algo config, merge with config_dict
- add all config to sacred, save to disk and run training 
```python
ex = Experiment("pymarl")
ex.add_config(config_dict)
ex.observers.append(FileStorageObserver.create(file_obs_path))
ex.run_commandline(params)
```
- inside main func, need to set seeds (numpy & torch) and then launch run framework
```python
np.random.seed(config["seed"])
th.manual_seed(config["seed"])
run(_run, config, _log)
```

### Run framework 
run 
- check config (set `use_cuda` if available), convert to args namespace
- get logger, log experiment parameters 
- set up tensorboard and sacred 
- `run_sequential`
- stop all threads and exit 
```python
 for t in threading.enumerate():
    if t.name != "MainThread":
        print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
        t.join(timeout=1)
        print("Thread joined")
os._exit(os.EX_OK)
```

run_sequential 
- args: args, logger 
- build runner (from `r_REGISTRY`)
- define `scheme`, `groups` and `preprocess` ???
```python
scheme = {
    "state": {"vshape": env_info["state_shape"]},
    "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
    "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
    "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
    "reward": {"vshape": (1,)},
    "terminated": {"vshape": (1,), "dtype": th.uint8},
}
groups = {
    "agents": args.n_agents
}
preprocess = {
    "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
}
```
- set up replay buffer
- set up multiagent controller (from `mac_REGISTRY`)
- set up learner (from `le_REGISTRY`)
- (optional), load model checkpoint to learner 
- while runner.t_env <= args.t_max
    - runner run to get `episode_batch`, dump to buffer 
    - learner train with `episode_sample` from buffer 
    - runner do test runs at test_interval
    - learner save model at save_interval
    - logger log stat at log_interval
- runner close env


evaluate_sequential
- args: args, runner 
- for in range(test_nepisode)
    - runner run 
- runner save replay 
- runner close env


### Arguments
- pymarl options: runner, mac, env, t_max, etc
- logging options: intervals, logging specs, folder paths
- RL hyperparams: learning rate, batch_size, gamma, etc 
- agent hyperparams: policy type, hidden dimensions
- experiment running params: id, label


### Miscellaneous 
- interesting way to get path name 
```python
# go up 2 level of folders from current file
os.path.join(dirname(dirname(abspath(__file__))), "results")
```


---
## Controller

BasicMAC
- init
    - args: scheme, groups, args
    - build agent, action_selector, hidden_states
- _get_input_shape
    - return obs shape from scheme
    - if args.obs_last_action, add actions_onehot shape
    - if args.obs_agent_id, add n_agents
- _build_agents 
    - instantiate agent from `agent_REGISTRY` and `input_shape` 
- _build_inputs 
    - takes in `batch`, `t`
    - assume homogeneous agents with flat observations, could delegate building inputs to each agent (if heterogeneous or non-flat observation)
- select_actions 
    - takes `ep_batch`, `t_ep`, `t_env`, `bs=slice(None)`, `test_mode=False`
    - get available actions, get action outputs from `forward`, pick actions from these 2 using action_selector 
    ```python
    avail_actions = ep_batch["avail_actions"][:, t_ep]
    agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
    chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
    ```
- forward 
    - takes in `ep_batch`, `t`, `test_mode=False`
    - build agent inputs, get agent outpus using `self.agent` from agent_inputs and hidden_states 
    - apply softmax if necessary
- other methods related to `self.agent`
    - parameters, load_state, cuda, save_models, load_models


---
## Modules 

### agents 
RNNAgent 
- fc1 (input_dim -> h_dim), rnn cell (h_dim -> h_dim), fc2 (h_dim -> n_actions)
- interesting way to init 0th hidden state (place to same device as model)
```python
self.fc1.weight.new(1, rnn_hidden_dim).zero_()
```
- forward, takes in `inputs`, `hidden_state`

### critics 
COMACritic
- get input shape from `scheme`, build MLP network (input_shape -> n_actions)
```
input_shape = state_shape + obs_shape + 2*num_agents*action_shape + num_agents
```
- forward, build inputs from `batch` & `t`, output `q`
- _build_inputs
    - 
    - output shape []

### mixers 
VDNMixer
- forward, takes in `agent_qs`, return sum of agent_qs along num_agent dim

QMixer 
- state_dim is product of state_shape
- 3 fc layers for hypernet, hyper_w_1 (state_dim -> embed_dim*num_agents), hyper_w_final (state_dim -> embed_dim), hyper_b_1 (state_dim -> embed_dim); V (2-layer MLP with ReLU, state_dim -> 1)
- forward
    - takes in `agent_qs`, `states`
    - states -> w1, b1, w_final, v
    - x (agent_qs) -> elu(w1*x + b1) -> w_final*x + v


---
## Components

MultinomialActionSelector
- has a decay schedule for epsilon (greedy exploration)
- select_action
    - takes in `agent_inputs`, `avail_actions`, `t_env`, `test_mode`
    - step epsilon schedule (from t_env)
    - if test_mode or test_greedy -> max over available actions, otherwise sample from Categorical

EpsilonGreedyActionSelector
- decay schedule for epsilon 
- select_action
    - akes in `agent_inputs`, `avail_actions`, `t_env`, `test_mode`
    - step epsilon schedule (from t_env)
    - sample random action from Categorical (freom avail_actions), generate random number 
    ```python
    pick_random = (random_numbers < self.epsilon).long()
    picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
    ```

OneHot
- transform, takes in `tensor`, output one-hot converted tensor
```python
y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
y_onehot.scatter_(-1, tensor.long(), 1)
```

DecayThenFlatSchedule
- eval, takes in `T`, return new epsilon (exponential or linear decay)

EpisodeBatch 
- takes in `scheme`, `groups`, `batch_size`, `max_seq_length`, `data`, `preprocess`, `device`
- initialize **self.data** with **data.transition_data = {}** and **data.episode_data = {}**
- _setup_data
    - apply preprocess transforms on and update scheme fields (e.g. actions to one-hot)
    - iterate through fields in scheme, populate with zeros of `dtype` (for each field) and to the required `device`
        - if field has **group** (e.g. "agents": n_agents), `shape` = (groups[group], *vshape); else, `shape` = vshape
        - if **episode_const**, populate `episode_data` with zeros (batch_size, *shape); else populate `transition_data` with zeros (batch_size, max_seq_length, *shape)
- to, takes in `device` and put all transition_data & episode_data to device
- update
    - takes in `data`, `bs=slice(None)`, `ts=slice(None)`, `mark_filled=True`
    - for each item/(key,value) in `data`, choose if target is transition_data or episode_data, then update the key field in target with value (transformed/preprocessed if neccessary)
    ```python
    v = th.tensor(v, dtype=dtype, device=self.device)
    target[k][_slices] = v.view_as(target[k][_slices])
    ```
- getitem 
    - takes in `item`, can be string, list of strings, 
- _parse_slices


ReplayBuffer 
- subclass from EpisodeBatch 
- initialized with `buffer_size` (same as batch_size), `buffer_index`， `episodes_in_buffer`
- insert_episode_batch
    - takes in `ep_batch` (instance of EpisodeBatch)
    - update with ep_batch.data.transition_data/episode_data
    - cyclic update buffer_index 
    ```python
    self.buffer_index = (self.buffer_index + ep_batch.batch_size)
    # num of episodes in buffer, is buffer_index before buffer is filled; else full size (buffer_size)
    self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
    self.buffer_index = self.buffer_index % self.buffer_size
    ```
- can_sample: check if episodes in buffer is at least batch_size
- sample: takes in `batch_size`, uniformly sample 


--- 
## Runner

EpisodeRunner
- init
    - takes in `args` and `logger`
    - build env (from `env_REGISTRY`), initialize `t_env=0`, `train_returns=[]`, `test_returns=[]`, `train_stats={}`, `test_stats={}`
- setup, regiter `new_batch` (as partial function to instantiate `EpisodeBatch`) and register `mac` 
- reset, re-initialize batch, reset env and reset time `t` to 0
- _log
    - takes in `returns`, `stats`, `prefix`
    - logger.log_stat on return_mean, return_std on step `t_env`, then clear returns
    - logger.log_stat on all items in stats, specifically item's mean over `n_episodes` on step `t_env`, then clear stats
- run 
    - reset first, intialize `terminated=False`, `episode_return=0`
    - while not terminated
        - setup before taking action 
        ```python
        pre_transition_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(pre_transition_data, ts=self.t)
        ```
        - select actions, step env, accumulate reward to episode_return, increment `t`
        ```python 
        post_transition_data = {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }
        self.batch.update(post_transition_data, ts=self.t)
        ```
    - batch update `last_data` (content same as pre_transition_data), update action in the last stored stste
    - collect and log returns and stats
    - return `self.batch`


ParallelRunner 
- Based (very) heavily on SubprocVecEnv from OpenAI Baselines <https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py>
- use multiprocessing (`Pipe`, `Process`), 
```python
# Make subprocesses for the envs
self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args)))) for worker_conn in self.worker_conns]
for p in self.ps:
    p.daemon = True
    p.start()
```
- multiprocessing in other methods can be controlled by interfacing with `self.parent_conns`, e.g.
```python
# Reset the envs
for parent_conn in self.parent_conns:
    parent_conn.send(("reset", None))
# Get stats back for each env
for parent_conn in self.parent_conns:
    parent_conn.send(("get_stats",None))
```

env_worker 
- takes in `remote`, `env_fn`
- ???


CloudpickleWrapper 
- use `cloundpickle`, serialize Python constructs not supported by the default pickle module, useful for cluster computing 
```python
def __init__(self, x):
    self.x = x
def __getstate__(self):
    import cloudpickle
    return cloudpickle.dumps(self.x)
def __setstate__(self, ob):
    import pickle
    self.x = pickle.loads(ob)
```


---
## Learner 

QLearner 
- init 
    - takes in `mac`, `scheme`, `logger`, `args`
    - instantiate `mixer`, optimizer and also targets (`target_mixer`, `target_mac`) by deepcopy
    - learnable parameters is the joint params from mac and mixer 
- _update_targets (hard updates)
```python
self.target_mac.load_state(self.mac)
if self.mixer is not None:
    self.target_mixer.load_state_dict(self.mixer.state_dict())
```
- other methods for mac (cuda, save_models, load_models)
- train 
    - takes in `batch`, `t_env`, `episode_num` (where batch is `EpisodeBatch`)
    - get batch rewards, actions, terminated, mask
    - calculate estimated Q values from mac and target mac 
    ```python
    mac_out = []
    self.mac.init_hidden(batch.batch_size)
    for t in range(batch.max_seq_length):
        agent_outs = self.mac.forward(batch, t=t)
        mac_out.append(agent_outs)
    mac_out = th.stack(mac_out, dim=1)  # Concat over time
    # Pick the Q-Values for the actions taken by each agent
    chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
    ```
    - calculate 1-step Q-Learning targets (get individual max Q values, mix if necessary)
    ```python 
    target_max_qvals = target_mac_out.max(dim=3)[0]
    # optionally mix
    if self.mixer is not None:
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
    # 1-step target 
    targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
    ```
    - calculate loss 
    ```python 
    # TD error 
    td_error = (chosen_action_qvals - targets.detach())
    mask = mask.expand_as(td_error)
    # 0-out the targets that came from padded data
    masked_td_error = td_error * mask
    # Normal L2 loss, take mean over actual data
    loss = (masked_td_error ** 2).sum() / mask.sum()
    ```
    - optimizer step
    - if reach `target_update_interval`, update_targets
    - if reach `learner_log_interval`, logger.log_stat on **loss**, **grad_norm**, **td_error_abs**, **q_taken_mean**, **target_mean** 


COMALearner
- init 
    - initialize `critic_training_steps=0`, instantiate `critic` from (`COMACritic`) and `target_critic`
    - make agent_optimizer and critic_optimizer
- _upate_targets (hard update)
```python
self.target_critic.load_state_dict(self.critic.state_dict())
```
- _train_critic 
    - takes in `batch`, `rewards`, `terminated`, `actions`, `avail_actions`, `mask`, `bs`, `max_t`
    - build TD target Q values
    ```python 
    target_q_vals = self.target_critic(batch)[:, :]
    targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)
    # Calculate td-lambda targets
    targets = build_td_lambda_targets(rewards, terminated, mask, targets_taken, self.n_agents, self.args.gamma, self.args.td_lambda)
    ```
    - train critic on each episode time 
    ```python 
    for t in reversed(range(rewards.size(1))):
        mask_t = mask[:, t].expand(-1, self.n_agents)
        if mask_t.sum() == 0:
            continue

        # actual Q values from critic 
        q_t = self.critic(batch, t)
        q_vals[:, t] = q_t.view(bs, self.n_agents, self.n_actions)
        q_taken = th.gather(q_t, dim=3, index=actions[:, t:t+1]).squeeze(3).squeeze(1)
        targets_t = targets[:, t]

        td_error = (q_taken - targets_t.detach())
        # critic optimizer step
        # ...
    ```
    - runnin_log has fields： **critic_loss**, **critic_grad_norm**, **td_error_abs**, **target_mean**, **q_taken_mean**
    - 
- train 
    - get batch rewards, actions, terminated, mask 
    - take critic optimization step 
    ```python
    q_vals, critic_train_stats = self._train_critic(batch, rewards, terminated, actions, avail_actions, critic_mask, bs, max_t) # bs is batch size
    ```
    - policy gradient step
    ```python
    q_vals = q_vals.reshape(-1, self.n_actions)
    pi = mac_out.view(-1, self.n_actions)
    baseline = (pi * q_vals).sum(-1).detach()

    # Calculate policy grad with mask
    q_taken = th.gather(q_vals, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
    pi_taken = th.gather(pi, dim=1, index=actions.reshape(-1, 1)).squeeze(1)
    pi_taken[mask == 0] = 1.0
    log_pi_taken = th.log(pi_taken)

    advantages = (q_taken - baseline).detach()

    coma_loss = - ((advantages * log_pi_taken) * mask).sum() / mask.sum()
    ``` 
    - agent/actor optimizer step 
    - if reach `target_update_interval`, update_targets
    - if reach `learner_log_interval`, logger.log_stat on critic `running_log` and **advantage_mean**, **coma_loss**, **agent_grad_norm**, **pi_max**



---
## Utils 

### logging 
Logger 
- takes in `console_logger`, initialize with 
    ```python
    self.stats = defaultdict(lambda: [])
    ```
- setup_tb
    - takes in `directory_name`, lazy loading 
    ```python
    from tensorboard_logger import configure, log_value
    ```
- setup_sacred 
    - takes in `sacred_run_dict`, initialize sacred_info
- log_stat 
    - args: key, value, t, to_sacred
    - update statistics value with 
    ```python
    self.stats[key].append((t, value))
    ```
    - log to TensorBoard with tb_logger 
    - log to Sacred with sacred_info, each key has a list of values, also another field `<key>_T` as a list of time steps
- print_recent_stats 
    - log out items in `self.stats` to console 
    - stat is averaged over last 5 (window) steps 


get_logger 
- use `logging.getLogger` 
- return custom logger to console 


build_td_lambda_targets
- takes in `rewards`, `terminated`, `mask`, `target_qs`, `n_agents`, `gamma`, `td_lambda`
```python
# Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
# Initialise  last  lambda -return  for  not  terminated  episodes
ret = target_qs.new_zeros(*target_qs.shape)
ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
# Backwards  recursive  update  of the "forward  view"
for t in range(ret.shape[1] - 2, -1,  -1):
    ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
# Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
return ret[:, 0:-1]
```