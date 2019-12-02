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
