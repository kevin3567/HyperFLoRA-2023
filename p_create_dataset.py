import os
import pickle
import numpy as np
import torch
import itertools

from config.options import args_parser
from misc.data_fetch import fetch_data_w_rand_order


# Warning: in this implementation, it is possible that certain users can acquire mono-label shards (all samples
# belong to the same class).
# To avoid the probability of this happening, increase the shard_per_user and try different seeds. To ensure no
# mono-label users exist, ensure that the [UNRESOLVED] list printout is empty, before using the organized dataset
# for subsequent training.
def noniid(dict_dataset, num_users, shard_per_user, rand_set_all=[]):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # SAMPLE-TO-SHARD PROCESSING
    # shatter each class into shards to build the client dataset
    num_classes = len(dict_dataset)
    shard_per_class = int(shard_per_user * num_users / num_classes)  # reasonable only if dataset is balanced
    for label in dict_dataset.keys():  # for each unique label, do ...
        x = dict_dataset[label]  # get list of relevant samples by idx
        num_leftover = len(x) % shard_per_class  # get remainder given balanced generate-to=shard assignment
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))  # reorganize samples over shards_per_class
        x = list(x)  # convert from tensor to list, allow for variable generate count per shard

        for i, idx in enumerate(leftover):  # add a leftover sample to each shards
            x[i] = np.concatenate([x[i], [idx]])
        dict_dataset[label] = x

    # LABEL-TO-USER PROCESSING (if not given)
    # form shard-to-user mapping if it is empty
    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class  # repeat [1..num_classes] tiling by shard_per_classes
        np.random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))  # random assignment of shard to user
        # KNOWN ISSUE: be aware that sometimes, a user may grad mono-labels shards. If so, try a different seed.
        print("[UNRESOLVED] Dummy Users With Mono-Label Shards: {}".format(  # this should be empty
            [x for x in rand_set_all if np.all(x == x[0])]),
            flush=True)

    # SHARD-TO-USER PROCESSING (from label-to-user)
    # based on labels acquired by user, assign a random shard belonging to the label to the user;
    # sample (idxs) within the shard are thereby acquired by the user.
    for i in range(num_users):  # for each user, do ...
        rand_set_label = rand_set_all[i]  # label assigned to user
        rand_set = []
        for label in rand_set_label:  # for each label assigned to user, select one shard randomly
            # selected shard (and corresponding samples) are removed after selection
            idx = np.random.choice(len(dict_dataset[label]),
                                   replace=False)  # why replace if doing only single randchoice?
            rand_set.append(dict_dataset[label].pop(idx))  # append popped idx (as it is selected)
        dict_users[i] = np.concatenate(rand_set)  # store sample-to-user

    return dict_users, rand_set_all  # return sample-to-user dictionary, label-to-user dictionary


def organize_data_shard(args):
    # note that the data sample order (dict_tr_*) are pre-shuffled in fetch_data_w_rand_order
    dataset_tr, dataset_vl, dataset_te, dict_tr_train, dict_tr_valid, dict_te_test = fetch_data_w_rand_order(args)

    # whether dataset is iid
    if args.iid:
        exit("IID distribution is not implemented.")
        raise NotImplementedError
    else:
        # create test set based on rand_set_all label-to-user assignment
        users_tr_train, rand_set_all = noniid(dict_tr_train, args.num_users, args.shard_per_user)
        users_tr_valid, rand_set_all = noniid(dict_tr_valid, args.num_users, args.shard_per_user,
                                              rand_set_all=rand_set_all)
        users_te_test, rand_set_all = noniid(dict_te_test, args.num_users, args.shard_per_user,
                                             rand_set_all=rand_set_all)
        print("Sizes: Train - {}, Validation - {}, Test - {}".format(
            len(list(itertools.chain(*users_tr_train.values()))),
            len(list(itertools.chain(*users_tr_valid.values()))),
            len(list(itertools.chain(*users_te_test.values()))),
        ))

    return dataset_tr, dataset_vl, dataset_te, users_tr_train, users_tr_valid, users_te_test


if __name__ == '__main__':
    # parse args
    args = args_parser()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dir = './save/{}/exp_iid{}_K{}_C{}_shard{}_val{}/{}_sd{}'.format(
        args.dataset,
        args.iid,
        args.num_users,
        args.frac,
        args.shard_per_user,
        args.val_split,
        args.results_save,
        args.seed)
    if not os.path.exists(os.path.join(base_dir)):
        os.makedirs(os.path.join(base_dir), exist_ok=True)

    # load dataset
    dataset_tr, dataset_vl, dataset_te, \
    dict_users_train, dict_users_valid, dict_users_test = \
        organize_data_shard(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_valid, dict_users_test), handle)

    print("Done setting up dataset.")
