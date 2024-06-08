import os
import pickle
import numpy as np
import torch
import itertools

from config.options import args_parser
from misc.data_fetch import fetch_data


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
        x = dict_dataset[label]  # get list of relevant samples by uidx
        num_leftover = len(x) % shard_per_class  # get remainder given balanced generate-to=shard assignment
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))  # reorganize samples over shards_per_class
        x = list(x)  # convert from tensor to list, allow for variable generate count per shard

        for i, idx in enumerate(leftover):  # add a leftover samples to each shards
            x[i] = np.concatenate([x[i], [idx]])
        dict_dataset[label] = x

    # LABEL-TO-USER PROCESSING (if not given)
    if len(rand_set_all) == 0:  # form shard-to-_user mapping if it is empty
        rand_set_all = list(range(num_classes)) * shard_per_class  # repeat [1..num_classes] tiling by shard_per_classes
        np.random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))  # random assignment of shard to _user
        # UNRESOLVED
        print("[UNRESOLVED] Dummy Users With Mono-Label Shards: {}".format(
            [x for x in rand_set_all if np.all(x == x[0])]),
            flush=True)

    # SHARD-TO-USER PROCESSING (from label-to-_user)
    # divide and assign
    for i in range(num_users):  # for each _user, do ...
        rand_set_label = rand_set_all[i]  # label assigned to _user
        rand_set = []
        for label in rand_set_label:  # for each label assigned to _user, select one shard by random
            # shards uidx and corresponding sample_list are removed after selection
            idx = np.random.choice(len(dict_dataset[label]),
                                   replace=False)  # why replace if doing only single randchoice?
            rand_set.append(dict_dataset[label].pop(idx))  # append popped uidx (as it is selected)
        dict_users[i] = np.concatenate(rand_set)  # store shard-to-_user as a dict

    # TODO: Reset new generated dataset checks
    # test that the dataset is properly set up
    # test = []
    # for key, value in dict_users.items():  # for each _user, do checks
    #     x = np.unique(torch.tensor(dataset.targets)[value])
    #     assert(len(x)) <= shard_per_user  # no more that "shard_per_user" number of classes in each _user
    #     test.append(value)
    # test = np.concatenate(test)
    # assert(len(test) == len(dataset))  # all data samples are used
    # assert(len(set(list(test))) == len(dataset))  # no duplicates in idxs

    return dict_users, rand_set_all  # return shard-to-_user dictionary, label-to=_user dictionary


def organize_data_shard(args):  # TODO: add more datasets as needed here

    dataset_tr, dataset_vl, dataset_te, dict_tr_train, dict_tr_valid, dict_te_test = fetch_data(args)

    # whether dataset is iid
    if args.iid:
        exit("IID distribution is not implemented.")
    else:
        # create test set based on rand_set_all label-to-_user assignment
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
