#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # algorithm arguments
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user: Sh")
    parser.add_argument('--rounds', type=int, default=10, help="rounds of training: R")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--bs', type=int, default=128, help="test batch size: Bs")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--tg_lr', type=float, default=0.01, help="tgarch learning rate: tg_lr")
    parser.add_argument('--tg_lr_decay', type=float, default=1.0, help="tgarch learning rate decay: tg_lr_decay")
    parser.add_argument('--tg_momentum', type=float, default=0.5, help="tgarch tg_momentum: tg_momentum")
    parser.add_argument('--tg_g_clip', type=float, default=1., help="tgarch gradient max norm: tg_g_clip")
    parser.add_argument('--hyp_lr', type=float, default=0.01, help="hyparch SGD learning rate: hyp_lr")
    parser.add_argument('--hyp_lr_decay', type=float, default=1.0, help="hyparch learning rate decay: hyp_lr_decay")
    parser.add_argument('--hyp_momentum', type=float, default=0.9, help="hyparch tg_momentum: hyp_tg_momentum")
    parser.add_argument('--hyp_wd', type=float, default=1e-3, help="hyparch weight decay: hyp_wd")
    parser.add_argument('--hyp_g_clip', type=float, default=50., help="hyparch gradient max norm: hyp_g_clip")
    parser.add_argument('--user_tr_interval', type=float, default=100, help="interval between user tgmodel retrain")

    # experiment arguments
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--val_split', type=float, default=0.1, help="proportion of train data used for validation")
    parser.add_argument('--users_split_ratio', type=float, default=1.0, help="number of users participating in "
                                                                            "training: Kp")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--load_ckpt', type=str, default='', help='define pretrained federated model path, must '
                                                                  'specify pathing starting from experiment folder '
                                                                  'run_*_sd*/ (prevents data leakage).')
    parser.add_argument('--results_save', type=str, default='dummy', help='define results save folder')
    parser.add_argument('--val_interval', type=int, default=1, help='how often to assess on val set')
    parser.add_argument('--save_interval', type=int, default=50, help='frequency of model saving')
    parser.add_argument('--do_debug', action="store_true", default=False, help='run ind debug mode')

    args = parser.parse_args()
    return args
