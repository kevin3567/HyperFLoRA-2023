This is the code for the paper HyperFLoRA: Federated Learning with 
Instantaneous Personalization (https://epubs.siam.org/doi/10.1137/1.
9781611978032.94)

For cifar10 experiment:
- to generate data partition:
    `nohup bash run_cifar10/do_create_dataset_cifar10.sh > lcheck.log`
- to pretrain initial model:
    `nohup bash run_cifar10/do_train_vitbasic_cifar10.sh > lcheck.log`
- to do hyperflora training:
    `nohup bash run_cifar10/do_train_vithyperflora_cifar10.sh > lcheck.log`

For cifar100 experiment:
- to generate data partition:
    `nohup bash run_cifar100/do_create_dataset_cifar100.sh > lcheck.log`
- to pretrain initial model:
    `nohup bash run_cifar100/do_train_vitbasic_cifar100.sh > lcheck.log`
- to do hyperflora training:
    `nohup bash run_cifar100/do_train_vithyperflora_cifar100.sh > lcheck.log`

Note that:
- When running `p_create_dataset.py`, it is possible (unlikely) that certain 
  users can acquire mono-label shards (all samples in the user have the same 
  label). To avoid this, make sure that the following printout is observed: 
  `[UNRESOLVED] Dummy Users With Mono-Label Shards: []` (the list should be 
  empty).
- In `p_train_vitbasic.py`, central refers to the global models. 
  user refers to the local model.
- In `p_train_vithyperflora.py`, there are three type of models: user (user), 
  hypernet (hyp), target (tg). (Note that user model trainable, as it only 
  outputs a fixed client representation vector.)
- There are three type of dataset: train, valid(ation), test. Note that 
  train and validation sets are partitioned from a common set, 
  so their sample indices should be disjoint. Test set is drawn from a 
  separate set.
- There are two types of users: participant (does local training) and 
  bystander (does no training). Designation follows that the first 80% 
  (adjustable) of users are participants, and the latter are bystander. This 
  is for consistency between experiments. Dataset alloted to each user is 
  determined by `p_create_dataset.py`, but should all be disjoint.
- The term "pseudo" refers to object/process conducted within a 
  pseudo-client (formed by pairing two users, and alternating LoRA training 
  between them)
- In `p_train_vitbasic.py` and `p_train_vithyperflora.py`, variables after 
  `if __name__ == "__main__"` starts with "_" to prevent accidental shadowing 
  of variables within declared functions.
- In general, training outputs are shown as printout. So using `nohup` and a 
  log file to acquire the printout information is encouraged.
- To get the best model from the experiment, look for the model file 
  with the `best_` tag. For example the best model acquired after running 
  `do_train_vitbasic_cifar100.sh` should be 
  `train_vitbasic_users_split_ratio_0.8/best_20000.pt`.
