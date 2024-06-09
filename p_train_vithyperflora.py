import copy
import time
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config.options import args_parser
from misc.data_fetch import fetch_data_w_rand_order, DatasetSplit
from misc.user_local import LocalTrainer_HN, LocalTester_HN
import os

from config.hypparam_ViTLora_HypNet import create_architectures
from misc.helper_functs import (get_classes2indicator,
                                get_class2sample_dict,
                                get_param_count,
                                get_model_info,
                                set_pretr_weights,
                                check_model_gradreq,
                                split_users,
                                sample_users,
                                sample_user_pairs,
                                transfer_weights,
                                eval_all_users,
                                process_result)

EXP_TYPE_STUB = "train_vithyperflora"
# True: for investigation, reproducibility is more important
# False: for result collection, speed is more important
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def create_users(num_users, userrep_list, usermodel, tgmodel):
    usermodel_list = []
    tgmodel_list = []
    for uidx in range(num_users):
        # copy tgarch (with pretrained weights)
        tgmodel_list.append(copy.deepcopy(tgmodel))  # user model
        # copy userarch (with appropriate initialization)
        local_usermodel = copy.deepcopy(usermodel)
        local_usermodel.assign_param(userrep_list[uidx])  # assigned weights should be frozen
        usermodel_list.append(local_usermodel)  # user representation (a model that returns a fixed rep vector)
    return usermodel_list, tgmodel_list


def set_tgmodel_gradreq(tgmodel, hypmodel):
    w_tgmodel_keys = [x for x in tgmodel.state_dict().keys()]
    w_tgmodel_assignable_keys = hypmodel.get_tgweight_keys()
    # freeze all keys not assignable by the hypernetwork
    w_tgmodel_frozen_keys = list(set([x for x in w_tgmodel_keys if
                                      x.startswith(("network.",))]).difference(w_tgmodel_assignable_keys))
    for key_name, param in tgmodel.named_parameters():
        if key_name in w_tgmodel_frozen_keys:
            param.requires_grad = False
        elif key_name in w_tgmodel_assignable_keys:
            param.requires_grad = True
        else:
            assert False, "Key {} is not found in TargetModel".format(key_name)


def run_evaluation(net_description,  # hyparch can be fed None to evaluate the loaded pretrained model
                   hyparch,
                   all_tgarch_list,
                   all_userarch_list,
                   dataset_eval,
                   dict_users_eval,
                   idxs_user_part,
                   idxs_user_byst,
                   args):
    with torch.no_grad():
        if hyparch is not None:
            hyparch.eval()
        for uidx in range(args.num_users):
            net_local = all_tgarch_list[uidx]
            if hyparch is not None or args.do_debug:
                # set the hyparch generated weights
                _ = generate_weights(userrep=all_userarch_list[uidx](),
                                     tgmodel=net_local,
                                     hypmodel=hyparch)
            net_local.eval()

        acc_eval_local_list, loss_eval_local_list = eval_all_users(net_list=all_tgarch_list,
                                                                   dataset_eval=dataset_eval,
                                                                   dict_users_eval=dict_users_eval,
                                                                   num_users=args.num_users,
                                                                   batch_size=args.bs,
                                                                   device=args.device,
                                                                   return_all=True)
        (acc_eval_loc_part_mean, acc_eval_loc_byst_mean, acc_eval_loc_all_mean), \
        (acc_eval_loc_part_std, acc_eval_loc_byst_std, acc_eval_loc_all_std), \
        (loss_eval_loc_part_mean, loss_eval_loc_byst_mean, loss_eval_loc_all_mean) = \
            process_result(acc_list=acc_eval_local_list,
                           loss_list=loss_eval_local_list,
                           idxs_part=idxs_user_part,
                           idxs_byst=idxs_user_byst)
        print("Model with {}, "
              "Average Participant Eval Accuracy/Loss: {:.2f}(w/{:.2f})/{:.3f}, "
              "Average Bystander Eval Accuracy/Loss: {:.2f}(w/{:.2f})/{:.3f}, "
              "Average All Eval Accuracy/Loss: {:.2f}(w/{:.2f})/{:.3f}, ".format(
            net_description,
            acc_eval_loc_part_mean, acc_eval_loc_part_std, loss_eval_loc_part_mean,
            acc_eval_loc_byst_mean, acc_eval_loc_byst_std, loss_eval_loc_byst_mean,
            acc_eval_loc_all_mean, acc_eval_loc_all_std, loss_eval_loc_all_mean),
            flush=True)
    return acc_eval_loc_part_mean, acc_eval_loc_part_std, loss_eval_loc_part_mean, \
           acc_eval_loc_byst_mean, acc_eval_loc_byst_std, loss_eval_loc_byst_mean, \
           acc_eval_loc_all_mean, acc_eval_loc_all_std, loss_eval_loc_all_mean


def generate_weights(userrep, tgmodel, hypmodel):
    # set the tgmodel with the hypmodel generated weights
    w_tgmodel = tgmodel.state_dict()
    w_tgmodel_assignable_dict = hypmodel(userrep)[0]
    for k, v in w_tgmodel_assignable_dict.items():
        w_tgmodel[k] = v
    tgmodel.load_state_dict(w_tgmodel)
    return w_tgmodel_assignable_dict


def train_hypnet_standard(user_train_lr, dataset_tr, sample_idxs_tr,
                          hypmodel, tgmodel, user_rep, args):
    # w_tglocal_assignable contains only the weights assignable by hypernetwork,
    # w_tgmodel_init/fin contains all weights within tgmodel

    local = LocalTrainer_HN(dataset_tr=dataset_tr,
                            idxs_tr=sample_idxs_tr,
                            local_bs=args.local_bs)

    w_tglocal_assignable = generate_weights(userrep=user_rep,
                                            tgmodel=tgmodel,
                                            hypmodel=hypmodel)
    tgmodel.train()
    w_tgmodel_init = copy.deepcopy(tgmodel.state_dict())  # Fix the initial weights

    w_tgmodel_fin, *_ = local.do_train(net=tgmodel.to(args.device),
                                       lr=user_train_lr,
                                       momentum=args.tg_momentum,
                                       local_ep=args.local_ep,
                                       grad_clip=args.tg_g_clip,
                                       device=args.device)

    # compute the gradient between tgarch weights (where assignable by hyperarch)
    # hyparch generates w_tgmodel_init values, and learns to approximate w_tgmodel_fin
    w_tgmodel_delta = OrderedDict(
        {k: w_tgmodel_init[k] - w_tgmodel_fin[k] for k in hypmodel.get_tgweight_keys()}
    )
    hypmodel_grad = torch.autograd.grad(
        outputs=list(w_tglocal_assignable.values()),  # this is the output of the hypernetwork
        inputs=_hyparch.parameters(),  # variables for which gradients are taken wrt
        grad_outputs=list(w_tgmodel_delta.values())
    )
    return hypmodel_grad


def form_psuser(usermodel_pair, class2samples_train_pair, args):
    assert len(usermodel_pair) == 2 and len(class2samples_train_pair) == 2, \
        "usermodel_pair and class2samples_train_pair should have a length of 2"
    classes_avail = torch.cat([usermodel().nonzero()[:, 1] for usermodel in usermodel_pair], dim=0)
    num_classes_avail = classes_avail.size(0)
    classes_chose = classes_avail[torch.randperm(num_classes_avail)][torch.div(num_classes_avail, 2).ceil().int():]
    psrep = torch.zeros_like(usermodel_pair[0]())  # create a dummy pseudo_rep
    psrep[:, classes_chose] = 1  # fill the pseudo_rep
    psusers_train_pair = []
    for c2s_tr_user in class2samples_train_pair:
        # get all samples in client uidx that belongs to a class marked by the pseudo_rep
        # for-if could be made more efficient by using set intersection
        psusers_train = []
        for label in classes_chose.cpu().numpy():
            if label in c2s_tr_user:
                psusers_train += c2s_tr_user[label]
        psusers_train_pair.append(np.array(psusers_train))
    assert len(psusers_train_pair) == 2, "psusers_train_pair should have a length of 2"
    return classes_chose, psrep, psusers_train_pair


def train_hypnet_paired(user_train_lr,
                        dataset_tr, sample_idxs_tr_pair,
                        hypmodel, tgmodel_pair, pseudouser_rep,
                        args):
    # two LOCAL_Trainer_HN (list) to alternate net_local training
    # create LocalTrainer only if there are actually data
    assert len(sample_idxs_tr_pair) == 2 and len(tgmodel_pair) == 2, \
        "usermodel_pair and class2samples_train_pair should have a length of 2"
    localtrainer_list = [LocalTrainer_HN(dataset_tr=dataset_tr,
                                         idxs_tr=sample_idxs_tr_pair[loc_uidx],
                                         local_bs=args.local_bs) for
                         loc_uidx in range(len(tgmodel_pair)) if
                         (len(sample_idxs_tr_pair[loc_uidx]) > 0)]

    w_tgcomb_assignable_dict = generate_weights(userrep=pseudouser_rep,
                                                tgmodel=tgmodel_pair[0],
                                                hypmodel=hypmodel)
    w_tgmodel_init = copy.deepcopy(tgmodel_pair[0].state_dict())
    w_tgmodel_fin = None  # make sure model actually undergoes training
    for ep in range(args.local_ep):  # alternated training keep trakc of comm cost
        for loc_uidx in range(len(localtrainer_list)):
            w_tgmodel_fin, *_ = localtrainer_list[loc_uidx].do_train(net=tgmodel_pair[loc_uidx],
                                                                     lr=user_train_lr,
                                                                     momentum=args.tg_momentum,
                                                                     local_ep=1,
                                                                     grad_clip=args.tg_g_clip,
                                                                     device=args.device)
            transfer_weights(weight_keys=w_tgcomb_assignable_dict.keys(),
                             src_model=tgmodel_pair[loc_uidx],
                             tgt_model=tgmodel_pair[(loc_uidx + 1) % len(tgmodel_pair)])

    w_tgmodel_delta = OrderedDict(
        {k: w_tgmodel_init[k] - w_tgmodel_fin[k] for k in hypmodel.get_tgweight_keys()}
    )
    hypmodel_grads = torch.autograd.grad(
        list(w_tgcomb_assignable_dict.values()),
        hypmodel.parameters(),
        grad_outputs=list(w_tgmodel_delta.values())
    )
    return hypmodel_grads


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
    base_dir = './save/{}/exp_iid{}_K{}_C{}_shard{}_val{}/{}_sd{}'.format(
        _args.dataset,
        _args.iid,
        _args.num_users,
        _args.frac,
        _args.shard_per_user,
        _args.val_split,
        _args.results_save,
        _args.seed)
    EXP_TYPE_STUB += '_users_split_ratio_{}'.format(_args.users_split_ratio)
    if not os.path.exists(os.path.join(base_dir, EXP_TYPE_STUB)):
        os.makedirs(os.path.join(base_dir, EXP_TYPE_STUB), exist_ok=True)
    # define results file and variable
    results_save_path = os.path.join(base_dir, EXP_TYPE_STUB, 'results.csv')

    # LOAD DATASET
    _dataset_train, _dataset_valid, _dataset_test, *_ = fetch_data_w_rand_order(_args)
    _data_info_path = os.path.join(base_dir, 'dict_users.pkl')
    # note that train and valid set should have no common indices. Test set is taken from a different partition.
    with open(_data_info_path, 'rb') as handle:
        _dict_users_train, _dict_users_valid, _dict_users_test = pickle.load(handle)
    assert all([len(v) > 0 for k, v in _dict_users_valid.items()]), \
        "(Assertion) Must have validation set for each user."
    _class2samples_train = get_class2sample_dict(dataset=_dataset_train, dict_users=_dict_users_train)
    _class2samples_valid = get_class2sample_dict(dataset=_dataset_valid, dict_users=_dict_users_valid)

    # GET MODEL INFO
    # instantiate usermodel, tgmodel and hypmodel
    _userarch, _tgarch, _hyparch = create_architectures(num_classes=_args.num_classes)
    _userarch.to(_args.device)
    _tgarch.to(_args.device)
    _hyparch.to(_args.device)

    # GET INFO FROM EACH ARCH
    _w_userarch_dict, _w_userarch_keys = get_model_info(model=_userarch, modelname="UserArch (userarch)")
    _w_tgarch_dict, _w_tgarch_keys = get_model_info(model=_tgarch, modelname="TargetArch (tgarch)")
    _w_hyparch_dict, _w_hyparch_keys = get_model_info(model=_hyparch, modelname="HyperArch (hyparch)")
    del _w_userarch_dict, _w_tgarch_dict, _w_hyparch_dict

    # INITIALIZE TGARCH
    # load pretrained tgmodel weights (acquired from initial fedavg training)
    if _args.load_ckpt != "":
        ckpt_path = os.path.join(base_dir, _args.load_ckpt)
        set_pretr_weights(model=_tgarch, ckpt_path=ckpt_path)
        print("Model Init: Loaded data from {}".format(ckpt_path))
    else:
        print("Model Init: Training from scratch")

    # SET GRAD_REQ FOR TGARCH
    set_tgmodel_gradreq(tgmodel=_tgarch, hypmodel=_hyparch)
    check_model_gradreq(model=_userarch, modelname="UserArch (userarch)")
    check_model_gradreq(model=_tgarch, modelname="TargetArch (tgarch)")
    check_model_gradreq(model=_hyparch, modelname="HyperArch (hyparch)")

    # CREATE USER MODEL (VECTOR REP) AND TARGET MODEL FOR EACH USER
    # iterate through the validation set (not test set) of each user to obtain the representation vector
    _user_label_indicators = {k: get_classes2indicator([_dataset_valid[i][1] for i in v], _args.num_classes) for
                              k, v in _dict_users_valid.items()}
    _all_userarch_list, _all_tgarch_list = create_users(num_users=_args.num_users,
                                                        userrep_list=_user_label_indicators,
                                                        usermodel=_userarch,
                                                        tgmodel=_tgarch)

    # DEFINE PARTICIPANT AND BYSTANDER USERS
    # first X users are participant, the remaining are bystander (keeps the split consistent)
    (_idxs_user_part, _part_num_users), (_idxs_user_byst, _byst_num_users) = \
        split_users(num_users=_args.num_users, split_ratio=_args.users_split_ratio)
    print("Participating Users (Num {}) Idx: {}".format(_part_num_users, _idxs_user_part))
    print("Bystander Users (Num {}) Idx: {}".format(_byst_num_users, _idxs_user_byst))

    # TEST PRETRAINED MODEL IN CONTEXT OF CURRENT PFL SCENARIO
    # TODO: create a model assessment function, as the code is repeated at the start and end.
    # set local _user models with initial hyperparameter
    _ = run_evaluation(net_description="Initial Weights (Test)",
                       hyparch=None,
                       all_tgarch_list=_all_tgarch_list,
                       all_userarch_list=_all_userarch_list,
                       dataset_eval=_dataset_test,
                       dict_users_eval=_dict_users_test,
                       idxs_user_part=_idxs_user_part,
                       idxs_user_byst=_idxs_user_byst,
                       args=_args,
                       )

    # EXPERIMENT INITIALIZATION
    _results = []  # stores assessed results
    # declare best checkpoint variables
    _best_hyparch, _best_acc, _best_epoch, = None, None, None
    # create misc variables
    _tg_lr_curr = _args.tg_lr  # set initial tg_lr, may decay; is used to create new optimizer in LocalTrainer
    _hyp_lr_curr = _args.hyp_lr

    # CREATE HYPARCH OPTIMIZER
    _hyp_optimizer = torch.optim.SGD(params=_hyparch.parameters(),
                                     lr=_hyp_lr_curr,
                                     momentum=_args.hyp_momentum,
                                     weight_decay=_args.hyp_wd)
    for _round in range(_args.rounds):
        # setup
        _round_start = time.time()
        _hyparch.train()
        _hyp_optimizer.zero_grad()

        # commence hyparch training
        _idxs_users_sel = sample_users(idxs_user_part=_idxs_user_part, frac=_args.frac)

        if _args.do_debug:
            print("Round {}, tg_lr_curr: {:.6f}, hyp_lr_curr: {:.6f}, {}".format(_round,
                                                                                 _tg_lr_curr,
                                                                                 _hyp_lr_curr,
                                                                                 _idxs_users_sel))
        # HYPERARCH TRAINING: PHASE 1
        ## do train standard
        for _uidx in _idxs_users_sel:  # train one client at a time (can we parallelize it)?
            _hypmodel_grad = \
                train_hypnet_standard(user_train_lr=_tg_lr_curr,
                                      dataset_tr=_dataset_train,
                                      sample_idxs_tr=_dict_users_train[_uidx],
                                      hypmodel=_hyparch,
                                      tgmodel=_all_tgarch_list[_uidx],
                                      user_rep=_all_userarch_list[_uidx](),
                                      args=_args)
            for _p, _g in zip(_hyparch.parameters(), _hypmodel_grad):
                _p.grad = _g if (_p.grad is None) else (_p.grad + _g)

        # HYPERARCH TRAINING: PHASE 2
        # sample client pair to create pseudo _user
        _idxs_userpairs_sel = sample_user_pairs(idxs_users_sel=_idxs_users_sel)
        for _userpair_idx in _idxs_userpairs_sel:  # assume is group for future development
            # build pseudo _user (client pair) dataset
            _classes_chose, _pseudouser_rep, _psusers_train_pair = form_psuser(
                usermodel_pair=[_all_userarch_list[_uidx] for _uidx in _userpair_idx],
                class2samples_train_pair=[_class2samples_train[_uidx] for _uidx in _userpair_idx],
                args=_args)

            # train psuedo user (client pair)
            _hypmodel_grad = \
                train_hypnet_paired(user_train_lr=_tg_lr_curr,
                                    dataset_tr=_dataset_train,
                                    sample_idxs_tr_pair=_psusers_train_pair,
                                    hypmodel=_hyparch,
                                    tgmodel_pair=[_all_tgarch_list[_uidx] for _uidx in _userpair_idx],
                                    pseudouser_rep=_pseudouser_rep,
                                    args=_args)

            # function ends here
            for _p, _g in zip(_hyparch.parameters(), _hypmodel_grad):
                _p.grad = _g if (_p.grad is None) else (_p.grad + _g)

        for _p in _hyparch.parameters():
            _p.grad /= (len(_idxs_users_sel) + len(_idxs_userpairs_sel))  # average over all gradients

        torch.nn.utils.clip_grad_norm_(_hyparch.parameters(), _args.hyp_g_clip)
        _hyp_optimizer.step()
        _hyp_optimizer.zero_grad()  # just in case it is not called at start of for loop

        # do tg_lr_curr decay after each comm round
        _tg_lr_curr *= _args.tg_lr_decay
        _hyp_lr_curr *= _args.hyp_lr_decay
        for _g in _hyp_optimizer.param_groups:
            _g['lr'] = _hyp_lr_curr

        # DO VALIDATION (direct avg-acc over all users)
        if (_round + 1) % _args.val_interval == 0:
            # copy weights of each target model weight to avoid potential disruption of training
            _acc_val_loc_part_mean, _acc_val_loc_part_std, _loss_val_loc_part_mean, *_ = \
                run_evaluation(net_description="Weights on Epoch {} (Val)".format(_round),
                               hyparch=_hyparch,
                               all_tgarch_list=_all_tgarch_list,
                               all_userarch_list=_all_userarch_list,
                               dataset_eval=_dataset_valid,
                               dict_users_eval=_dict_users_valid,
                               idxs_user_part=_idxs_user_part,
                               idxs_user_byst=_idxs_user_byst,
                               args=_args)

            if _best_acc is None or _acc_val_loc_part_mean > _best_acc:  # checkpointing decided by part-val-acc
                _best_hyparch = copy.deepcopy(_hyparch)
                _best_acc = _acc_val_loc_part_mean
                _best_epoch = _round

            # finalize results
            _results.append(np.array([_round, _loss_val_loc_part_mean, _acc_val_loc_part_mean, _best_acc]))
            _final_results = np.array(_results)
            _final_results = pd.DataFrame(_final_results,
                                          columns=['epoch', 'loss_val', 'acc_val', 'best_acc'])
            _final_results.to_csv(results_save_path, index=False)

        if (_round + 1) % _args.save_interval == 0:
            # save current round hyparch, in case training crashes
            _model_save_path = os.path.join(base_dir, EXP_TYPE_STUB, 'ckpt_{}.pt'.format(_round + 1))
            torch.save((_hyparch.state_dict()), _model_save_path)
            # save current best hyparch, to find the best model for down the line testing
            _best_save_path = os.path.join(base_dir, EXP_TYPE_STUB, 'best_{}.pt'.format(_round + 1))
            torch.save(_best_hyparch.state_dict(), _best_save_path)

        _round_end = time.time()
        if _args.do_debug:
            print("Round {} done. Time Taken: {}".format(_round, _round_end - _round_start))

    print('Done Training. Best Iteration: {} with Valid Accuracy: {}'.format(_best_epoch, _best_acc))

    # DO TESTING
    print("Testing the network on test set.", flush=True)
    _ckpt_types = {"Latest Weights (Test)": _hyparch,
                   "Best Weights (Test)": _best_hyparch}

    for _net_desc, _hyparch_fin in _ckpt_types.items():
        _ = run_evaluation(net_description=_net_desc,
                           hyparch=_hyparch_fin,
                           all_tgarch_list=_all_tgarch_list,
                           all_userarch_list=_all_userarch_list,
                           dataset_eval=_dataset_test,
                           dict_users_eval=_dict_users_test,
                           idxs_user_part=_idxs_user_part,
                           idxs_user_byst=_idxs_user_byst,
                           args=_args)

    _end = time.time()
    print("Done All. TIme Taken {}".format(_end - _start))
