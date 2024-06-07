import torch
import numpy as np
import copy
from misc.user_local import LocalTester_HN, LocalTrainer_HN


def get_classes2indicator(classes_list, num_classes):
    """Generate indicator vector from dataset_tr shards"""
    classes_uniq = torch.LongTensor(np.unique(classes_list))
    class_indicator = torch.zeros(1, num_classes)
    class_indicator[0, classes_uniq] = 1
    return class_indicator


def get_class2sample_dict(dataset, dict_users):
    class2samples = {}
    for uidx, sidx_list in dict_users.items():
        class2samples[uidx] = {}
        for sidx in sidx_list:
            sample_class = dataset[sidx][1]
            if sample_class in class2samples[uidx]:  # subsequent encounter of known class
                class2samples[uidx][sample_class].append(sidx)
            else:  # first encounter of new class
                class2samples[uidx][sample_class] = [sidx]
    return class2samples


def get_param_count(module):  # check
    """Get parameter count of module"""
    param_ct = 0
    for key in module.state_dict().keys():
        param_ct += module.state_dict()[key].numel()
    return param_ct


def get_model_info(model, modelname):
    w_model_dict = model.state_dict()
    w_model_keys = [x for x in w_model_dict.keys()]
    print("{} Weight Key Set".format(modelname))
    for key in w_model_keys:
        print("--" + key + ":" + str(model.state_dict()[key].numel()))
    print("{} Design: ".format(modelname))
    print(model)
    print("{} Total Parameters: {}".format(modelname, get_param_count(model)), flush=True)
    return w_model_dict, w_model_keys


def set_pretr_weights(model, ckpt_path):
    w_model_dict = model.state_dict()
    w_model_keys = [x for x in w_model_dict.keys()]

    w_model_pretr = torch.load(ckpt_path)
    for key_name in w_model_keys:
        if key_name.startswith("network."):
            weights = w_model_pretr.get(key_name, None)
            if weights is not None:
                print("++Found and Load: {}".format(key_name))
                w_model_dict[key_name] = weights
            else:
                print("++Missing and Skipped: {}".format(key_name))
    model.load_state_dict(w_model_dict)


def check_model_gradreq(model, modelname):
    print("Check {} grad_req status.".format(modelname))
    for key_name, param in model.named_parameters():
        print("--{}: {}".format(key_name, param.requires_grad))


def split_users(num_users, split_ratio):
    left_num_users = int(num_users * split_ratio)
    idxs_user_left = list(range(left_num_users))
    right_num_users = num_users - left_num_users
    idxs_user_right = list(range(left_num_users, left_num_users + right_num_users))
    return (idxs_user_left, left_num_users), (idxs_user_right, right_num_users)


def sample_users(idxs_user_part, frac):
    m = max(int(frac * len(idxs_user_part)), 1)
    idxs_users_sel = np.random.choice(idxs_user_part, m, replace=False)
    return idxs_users_sel


def sample_user_pairs(idxs_users_sel):
    idx_shuf = np.random.permutation(len(idxs_users_sel))
    idxs_userpairs_sel = idxs_users_sel[idx_shuf].reshape(-1, 2)
    return idxs_userpairs_sel


def eval_all_users(net_list, dataset_eval, dict_users_eval, num_users, batch_size, device, return_all=True):
    assert len(dict_users_eval) == num_users, "The dict_users_eval should have a length of num_users"
    accuracy_list = -np.ones(num_users)
    test_loss_list = -np.ones(num_users)
    for uidx in range(num_users):
        local_tester = LocalTester_HN(dataset_eval, dict_users_eval[uidx], local_bs=batch_size)
        accuracy, test_loss = local_tester.do_test(net_list[uidx], device)  # net_list should be preset by hyparch
        accuracy_list[uidx] = accuracy
        test_loss_list[uidx] = test_loss
    if return_all:
        return accuracy_list, test_loss_list
    return accuracy_list.mean(), test_loss_list.mean()

def process_result(acc_list, loss_list, idxs_part, idxs_byst):
    acc_part_mean = -1 if len(idxs_part) == 0 else acc_list[idxs_part].mean()
    acc_byst_mean = -1 if len(idxs_byst) == 0 else acc_list[idxs_byst].mean()
    acc_all_mean = acc_list.mean()
    acc_part_std = -1 if len(idxs_part) == 0 else acc_list[idxs_part].std()
    acc_byst_std = -1 if len(idxs_byst) == 0 else acc_list[idxs_byst].std()
    acc_all_std = acc_list.std()
    loss_part_mean = -1 if len(idxs_part) == 0 else loss_list[idxs_part].mean()
    loss_byst_mean = -1 if len(idxs_byst) == 0 else loss_list[idxs_byst].mean()
    loss_all_mean = loss_list.mean()
    return (acc_part_mean, acc_byst_mean, acc_all_mean), \
           (acc_part_std, acc_byst_std, acc_all_std), \
           (loss_part_mean, loss_byst_mean, loss_all_mean)