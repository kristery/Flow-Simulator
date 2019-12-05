# Flow-Simulator

## Env Testing
```python
python3 -m sample --pg-type td3
```

## Env Setting
check out **flow/flow/envs** for detailed information
### Action space
  * one action for each intersection
  * range within 0 and 1

### Observation space
  * fully observable state space (TrafficLightGridEnv)
    * for vehicles:
      1. velocity
      2. distance from the next intersection
      3. the unique edge it is traveling on
    * for each traffic light:
      1. current state (the flowing direction)
      2. last changed time
      3. whether it's yellow

### Reward
  * large delay penalty
  * switch penalty


### Network Indexing

   01 23 45
11
10

9
8
 
7
6

### Notes
  * step function is at flow/envs/base.py

### Useful Functions
  * env.k.vehicle.get_speed(veh_id)
  * env.k.network.max_speed()
  * env.network.node_mapping: iteratively give node and edge
  * env.k.vehicle.get_edge(veh_id): current edge veh_id is at
  * env.k.vehicle.get_ids_by_edge(edge): gives ids
    * for k in env.k.network.get_edge_list()
