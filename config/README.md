Contains model hyperparameter (and creator) and experiment configuration.
 - hypparam_{XXX}.py, where {XXX} is the name of the model:
   - contains dict {k: v} to initialize a fully executable model.
   - contains creator function that returns a model with same configuration.
 - options_{XXX}.py, where {XXX} is the name of an experiment
   - contains argparse settings about the particular experiment.
   - argparse variable can be provided durin command line execution.

Done:
- configs\hypparam_LeNet_AIM_FCL.py

Pending.py:
- options_CIFAR10.py

