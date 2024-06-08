DONE:
- Comments are revised to make the code understandable.
- Training and evaluation methods are revised to reduce repeated code.
- Separate configuration files are readjusted.

TODO:
- There are a lot of reinventing the wheels, where already implemented 
  functions are re-implemented in this code.
- Client and server operations needs clearer demarcation in this simulation.
  Consider implementing distinct client and server classes.
- More refactoring is necessary to remove redundant code and variables.
 

# GENERAL SHORTHAND:
# There are three type of models: _user (_user), hyper (hyp), target (tg).
# Note that _user model is not trainable, as it only generates a fixed client representation here.
# The actual client model is the target model.
# There are three type of dataset: train, valid(ation), test
# There are two types of users: participant (does local training) and bystander (does no training)
# The term "local" refers to object/process within a trainer
# The term "pseudo" refers to object/process conducted within a pseudo-client (formed by pairing two users)
# all variable names in main starts with "_" to prevent accidental shadowing of variable name
# once training loop starts, all users before use must undergo hypernetwork weight assignment before use
# This is an FL simulation, thus user and servers are not clearly demarcated.