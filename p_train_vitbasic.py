#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import re

import torch
from torch.utils.data import DataLoader

from config.options import args_parser
from misc.data_fetch import fetch_data_w_rand_order, DatasetSplit
import os

from config.hypparam_ViTBasic import create_architectures

EXP_TYPE_STUB = "train_vitbasic"
# True: for investigation, reproducibility is more important
# False: for result collection, speed is more important
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from misc.user_local import LocalTrainer_HN
from misc.helper_functs import (sample_users,
                                get_param_count,
                                get_model_info,
                                set_pretr_weights,
                                transfer_weights,
                                check_model_gradreq,
                                split_users,
                                process_result,
                                eval_all_users)


# manually define the weights that should be frozen, public, and private.
def set_weight_mode(model, w_all_keys):  # this only produces lists of keys, it does not modify anything.
    w_pr_keys = []
    w_pb_keys = list(set([x for x in w_all_keys if x.startswith(("network.",))]). \
                     difference(w_pr_keys))
    w_frz_keys = list(set([x for x in w_all_keys if x.startswith(("network.",))]). \
                      difference(w_pr_keys).difference(w_pb_keys))
    print("Complete Key Set")
    for key in w_all_keys:
        print("--" + key + ":" + str(model.state_dict()[key].numel()))
    print("Private Key Set")
    for key in w_pr_keys:
        print("--" + key)
    print("Public Key Set")
    for key in w_pb_keys:
        print("--" + key)
    print("Frozen Key Set")
    for key in w_frz_keys:
        print("--" + key)
    # check that all keys are categorized as private, public, or frozen
    assert set(w_pr_keys + w_pb_keys + w_frz_keys).intersection(w_all_keys) == set(w_all_keys)
    return w_pr_keys, w_pb_keys, w_frz_keys


def set_tgmodel_gradreq(model, w_frz_keys):
    for key_name, param in model.named_parameters():
        if key_name in w_frz_keys:
            param.requires_grad = False
        else:
            param.requires_grad = True


def run_evaluation(net_description,
                   model_central,
                   model_user_list,
                   w_public_keys,
                   w_private_keys,  # unused here
                   w_frozen_keys,  # unused here
                   dataset_eval,
                   dict_users_eval,
                   idxs_user_part,
                   idxs_user_byst,
                   args):
    for idx in range(args.num_users):
        transfer_weights(weight_keys=w_public_keys,
                         src_model=model_central,
                         tgt_model=model_user_list[idx])
        model_user_list[idx].eval()
    # compute and aggregate user accuracy using all users through local model on local test samples
    with torch.no_grad():
        acc_eval_loc_list, loss_eval_loc_list = \
            eval_all_users(net_list=model_user_list,
                           dataset_eval=dataset_eval,
                           dict_users_eval=dict_users_eval,
                           num_users=args.num_users,
                           batch_size=args.bs,
                           device=args.device,
                           return_all=True)
        (acc_eval_loc_part_mean, acc_eval_loc_byst_mean, acc_eval_loc_all_mean), \
        (acc_eval_loc_part_std, acc_eval_loc_byst_std, acc_eval_loc_all_std), \
        (loss_eval_loc_part_mean, loss_eval_loc_byst_mean, loss_eval_loc_all_mean) = \
            process_result(acc_list=acc_eval_loc_list,
                           loss_list=loss_eval_loc_list,
                           idxs_part=idxs_user_part,
                           idxs_byst=idxs_user_byst)
        print("Model with {}, "
              "Average Participant Eval Accuracy(StdDev)/Loss: {:.2f}(w/{:.2f})/{:.3f}, "
              "Average Bystander Eval Accuracy(StdDev)/Loss: {:.2f}(w/{:.2f})/{:.3f}, "
              "Average All Eval Accuracy(StdDev)/Loss: {:.2f}(w/{:.2f})/{:.3f}, ".format(
            net_description,
            acc_eval_loc_part_mean, acc_eval_loc_part_std, loss_eval_loc_part_mean,
            acc_eval_loc_byst_mean, acc_eval_loc_byst_std, loss_eval_loc_byst_mean,
            acc_eval_loc_all_mean, acc_eval_loc_all_std, loss_eval_loc_all_mean),
            flush=True)
    return acc_eval_loc_part_mean, acc_eval_loc_part_std, loss_eval_loc_part_mean, \
           acc_eval_loc_byst_mean, acc_eval_loc_byst_std, loss_eval_loc_byst_mean, \
           acc_eval_loc_all_mean, acc_eval_loc_all_std, loss_eval_loc_all_mean


def do_train(user_train_lr, dataset_tr, sample_idxs_tr,
             net_global, net_user,
             w_public_keys, w_private_keys, w_frozen_keys,
             args):

    local = LocalTrainer_HN(dataset_tr=dataset_tr,
                            idxs_tr=sample_idxs_tr,
                            local_bs=args.local_bs)
    transfer_weights(weight_keys=w_public_keys,
                     src_model=net_global,
                     tgt_model=net_user)
    net_user.train()

    w_local, loss, sample_ct = local.do_train(net=net_user.to(args.device),
                                              lr=user_train_lr,
                                              momentum=args.tg_momentum,
                                              local_ep=args.local_ep,
                                              grad_clip=args.tg_g_clip,
                                              device=args.device)
    return w_local, loss, sample_ct


if __name__ == '__main__':
    _start = time.time()

    # PARSE ARGS
    _args = args_parser()
    print("Printing all arguments:")
    for arg in vars(_args):
        print(arg, getattr(_args, arg))

    # SET GPU
    _args.device = torch.device('cuda:{}'.format(_args.gpu) if torch.cuda.is_available() and _args.gpu != -1 else 'cpu')

    # RANDOM SEED
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_args.seed)

    # CREATE RESULTS FOLDER
    _base_dir = './save/{}/exp_iid{}_K{}_C{}_shard{}_val{}/{}_sd{}'.format(
        _args.dataset,
        _args.iid,
        _args.num_users,
        _args.frac,
        _args.shard_per_user,
        _args.val_split,
        _args.results_save,
        _args.seed)
    EXP_TYPE_STUB += '_users_split_ratio_{}'.format(_args.users_split_ratio)
    if not os.path.exists(os.path.join(_base_dir, EXP_TYPE_STUB)):
        os.makedirs(os.path.join(_base_dir, EXP_TYPE_STUB), exist_ok=True)
    _results_save_path = os.path.join(_base_dir, EXP_TYPE_STUB, 'results.csv')

    # LOAD DATASET
    _dataset_train, _dataset_valid, _dataset_test, *_ = fetch_data_w_rand_order(_args)
    _data_info_path = os.path.join(_base_dir, 'dict_users.pkl')
    with open(_data_info_path, 'rb') as handle:
        _dict_users_train, _dict_users_valid, _dict_users_test = pickle.load(handle)
    assert all([len(v) > 0 for k, v in _dict_users_valid.items()]), \
        "(Assertion) Must have validation set for each user."

    # GET MODEL INFO
    # define parameter type
    _net_central = create_architectures(num_classes=_args.num_classes).to(_args.device)
    _w_central_accum, _w_central_keys = get_model_info(_net_central, modelname="Net Central")

    # assign parameter update mode
    _w_private_keys, _w_public_keys, _w_frozen_keys = set_weight_mode(model=_net_central,
                                                                      w_all_keys=_w_central_keys)

    # initialize model
    if _args.load_ckpt != "":
        ckpt_path = os.path.join(_base_dir, _args.load_ckpt)
        set_pretr_weights(model=_net_central, ckpt_path=ckpt_path)
        print("Model Init: Loaded data from {}".format(ckpt_path))
    else:
        print("Model Init: Training from scratch")
    del _w_central_accum

    set_tgmodel_gradreq(model=_net_central, w_frz_keys=_w_frozen_keys)
    check_model_gradreq(model=_net_central, modelname="Net Central")

    # copy _net_central to each user
    _net_user_list = []
    for _user in range(_args.num_users):  # for each user, copy the loaded checkpoint
        _net_user_list.append(copy.deepcopy(_net_central))

    # define participants and bystander users
    # first X users are participant, the remaining latter are bystander
    (_idxs_user_part, _part_num_users), (_idxs_user_byst, _byst_num_users) = \
        split_users(num_users=_args.num_users, split_ratio=_args.users_split_ratio)
    print("Participating Users (Num {}) Idx: {}".format(_part_num_users, _idxs_user_part))
    print("Bystander Users (Num {}) Idx: {}".format(_byst_num_users, _idxs_user_byst))

    # TEST INIT MODEL
    _ = run_evaluation(net_description="Initial Weights (Test)",
                       model_central=_net_central,
                       model_user_list=_net_user_list,
                       w_public_keys=_w_public_keys,
                       w_private_keys=_w_private_keys,
                       w_frozen_keys=_w_frozen_keys,
                       dataset_eval=_dataset_test,
                       dict_users_eval=_dict_users_test,
                       idxs_user_part=_idxs_user_part,
                       idxs_user_byst=_idxs_user_byst,
                       args=_args)

    # EXPERIMENT INITIALIZATION
    _results = []
    _loss_train = []
    _net_central_best, _net_user_list_best, _best_acc, _best_epoch = None, None, None, None
    # create misc variables
    _tg_lr_curr = _args.tg_lr  # set initial _tg_lr_curr, may decay over time
    _w_central_accum = copy.deepcopy(_net_central.state_dict())

    for _round in range(_args.rounds):
        # setup
        _round_start = time.time()

        for k in _w_central_keys:
            _w_central_accum[k] = torch.zeros_like(_w_central_accum[k])
        _local_loss_list = []
        _total_sample_ct = 0

        _idxs_users_sel = sample_users(_idxs_user_part, _args.frac)
        if _args.do_debug:
            print("Round {}, _tg_lr_curr: {:.6f}, {}".format(_round, _tg_lr_curr, _idxs_users_sel))

        for _uidx in _idxs_users_sel:
            _w_local, _loss, _sample_ct = \
                do_train(user_train_lr=_tg_lr_curr,
                         dataset_tr=_dataset_train,
                         sample_idxs_tr=_dict_users_train[_uidx],
                         net_global=_net_central,
                         net_user=_net_user_list[_uidx],
                         w_public_keys=_w_public_keys,
                         w_private_keys=_w_private_keys,
                         w_frozen_keys=_w_frozen_keys,
                         args=_args)
            _local_loss_list.append(_loss)

            _total_sample_ct += _sample_ct
            for _k in _w_public_keys:
                _w_central_accum[_k] += _w_local[_k] * _sample_ct

        # average the accumulated weights, then update central model
        for _k in _w_public_keys:
            _w_central_accum[_k] = torch.div(_w_central_accum[_k], _total_sample_ct)
        _net_central.load_state_dict(_w_central_accum)

        # decay tg_lr after each comm round
        _tg_lr_curr *= _args.tg_lr_decay

        _loss_avg = sum(_local_loss_list) / len(_local_loss_list)
        _loss_train.append(_loss_avg)

        if (_round + 1) % _args.val_interval == 0:
            _acc_val_loc_part_mean, _acc_val_loc_part_std, _loss_val_loc_part_mean, *_ = \
                run_evaluation(net_description="Weights on Epoch {} (Val)".format(_round),
                               model_central=_net_central,
                               model_user_list=_net_user_list,
                               w_public_keys=_w_public_keys,
                               w_private_keys=_w_private_keys,
                               w_frozen_keys=_w_frozen_keys,
                               dataset_eval=_dataset_valid,
                               dict_users_eval=_dict_users_valid,
                               idxs_user_part=_idxs_user_part,
                               idxs_user_byst=_idxs_user_byst,
                               args=_args)

            if _best_acc is None or _acc_val_loc_part_mean > _best_acc:  # find best ckpt by part-val-acc
                _net_central_best = copy.deepcopy(_net_central)
                _net_user_list_best = copy.deepcopy(_net_user_list)
                _best_acc = _acc_val_loc_part_mean
                _best_epoch = _round

            _results.append(np.array([_round, _loss_avg, _loss_val_loc_part_mean, _acc_val_loc_part_mean, _best_acc]))
            _final_results = np.array(_results)
            _final_results = pd.DataFrame(_final_results,
                                          columns=['round',
                                                  'loss_avg_tr',
                                                  'loss_part_val',
                                                  'acc_part_val',
                                                  'best_part_acc_val'])
            _final_results.to_csv(_results_save_path, index=False)

        # save experiment state in set intervals
        if (_round + 1) % _args.save_interval == 0:
            _model_save_path = os.path.join(_base_dir, EXP_TYPE_STUB, 'ckpt_{}.pt'.format(_round + 1))
            torch.save(_net_central.state_dict(), _model_save_path)
            _best_save_path = os.path.join(_base_dir, EXP_TYPE_STUB, 'best_{}.pt'.format(_round + 1))
            torch.save(_net_central_best.state_dict(), _best_save_path)

        _round_end = time.time()
        if _args.do_debug:
            print("Round {} done. Time Taken: {}".format(_round, _round_end - _round_start))

    print('Best model, Iteration: {}, Best Average Participant Valid Accuracy: {}'.format(_best_epoch, _best_acc))

    # DO TESTING
    print("Testing the network on test set.", flush=True)
    _ckpt_types = {"Latest Weights (Test)": (_net_central, _net_user_list),
                   "Best Weights (Test)": (_net_central_best, _net_user_list_best)}

    for _net_desc, (_net_central_fin, _net_user_list_fin) in _ckpt_types.items():
        _ = run_evaluation(net_description=_net_desc,
                           model_central=_net_central_fin,
                           model_user_list=_net_user_list_fin,
                           w_public_keys=_w_public_keys,
                           w_private_keys=_w_private_keys,
                           w_frozen_keys=_w_frozen_keys,
                           dataset_eval=_dataset_test,
                           dict_users_eval=_dict_users_test,
                           idxs_user_part=_idxs_user_part,
                           idxs_user_byst=_idxs_user_byst,
                           args=_args)

    _end = time.time()
    print("Done All. TIme Taken {}".format(_end - _start))
