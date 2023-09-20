#!/usr/bin/env python
# coding:utf-8
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import pandas as pd
from calibration_in_da import sweep_on_calibration_methods
from utils import ECELoss, get_weight_union

    

def load_data(feature_path, type = 'source_train', pseudo_labels = False):
    features = np.load(feature_path + f'_{type}_feature.npy')
    logits = torch.from_numpy(np.load(feature_path + f'_{type}_output.npy'))
    labels = torch.from_numpy(np.load(feature_path + f'_{type}_label.npy')).long()

    if pseudo_labels:
        pseudo_labels = torch.from_numpy(np.load(feature_path + f'_{type}_pseudo_label.npy')).long()
        return features, logits, labels, pseudo_labels
    
    else:
        return features, logits, labels


def compute_ece(method, 
                dataset, 
                task, 
                source_val_acc, 
                target_acc, 
                root_dir='features/', 
                optimizer = 'brute', 
                folder = '.', 
                est=True):
    

    # Compute estimated Espilon according to formula
    best_eps = (source_val_acc - target_acc) / ((1+1/64)* source_val_acc - 1/64)
    acc_ratio = target_acc / source_val_acc

    if args.est:
        # load estimated acc
        try:
            est_target_acc = np.load(f"accuracy_estimation/{dataset}_{method}_{task}_est_acc.npy")[0]
            est_acc_ratio = est_target_acc / source_val_acc
        except:
            est_acc_ratio = None
        tmp_ratio = est_acc_ratio
    else:
        tmp_ratio = acc_ratio


    feature_path = os.path.join(folder, root_dir, dataset, method, dataset + '_' + method + '_' + task)
    output_name = method + "_" + task
    print(output_name)

    ##############################################
    ########### ------ Get Data ------ ###########
    ##############################################

    features_source_train, logits_source_train, labels_source_train = load_data(feature_path, type='source_train')
    features_target, logits_target, labels_target, pseudo_labels_target = load_data(feature_path, type='target', pseudo_labels = True)
    features_source_val, logits_source_val, labels_source_val = load_data(feature_path, type='source_val')


    ##############################################
    ######## ------- Calibrate -------- ##########
    ##############################################

    ece_criterion = ECELoss()
    source_train_ece = ece_criterion(logits_source_train, labels_source_train).item()
    source_val_ece = ece_criterion(logits_source_val, labels_source_val).item()
    print("source_train_ece: {:4f}".format(source_train_ece))
    print("source_val_ece: {:4f}".format(source_val_ece))

    df_stat = pd.DataFrame(columns = ['aux_data', 'cal_method', 'ece_on_target', 'ece_on_target_val', 'optimal_temp'])

    # "Method 1: Vanilla Model (before calibration)"
    vanilla_target_ece = ece_criterion(logits_target, labels_target).item()
    print("vanilla_target_ece: {:4f}".format(vanilla_target_ece))

    # "Baseline methods"    
    methods = [
        'Oracle','Oracle_ada',
        'TempScaling', 'CPCS', 'TransCal'
    ]


    UTDC_methods = [
        'UTDC',
        'UTDC_ada',                 
    ]

    methods = methods + UTDC_methods
    for i in range(10):
        weight = get_weight_union(features_source_train, features_target, features_source_val)
        results = sweep_on_calibration_methods(logits_source_val = logits_source_val, 
                                    labels_source_val = labels_source_val,
                                    logits_target=logits_target, 
                                    labels_target=labels_target,
                                    weight=None,
                                    optimizer = optimizer,
                                    methods=methods,
                                    acc_fix=tmp_ratio)

        print (results)




if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--dataset', type=str, default='office-home')
    parser.add_argument('--folder', type=str)
    parser.add_argument('--est', type=bool, default=False)

    args = parser.parse_args()

    all_methods = ['CDAN+E', 'DANN', 'DANN+E']
    if args.dataset == 'office-home':
        df = pd.read_csv('accuracy_analysis/office-home_accuracy_analysis.csv')
        tasks = ['R2A', 'R2C', 'R2P', 'A2R', 'A2C', 'A2P', 'C2A', 'C2R', 'C2P', 'P2A', 'P2R', 'P2C']

    for i, task in enumerate():
        for j, method in enumerate(all_methods):
            print (f'{task} - {method} ----->')
            entry = df[(df['method'] == method) & (df['task'] == task)]
            source_val_acc = entry['accuracy_source_val']
            target_acc = entry['accuracy_target']


            compute_ece(method = method, 
                        dataset = args.dataset, 
                        task=task, 
                        folder = args.folder,
                        source_val_acc = source_val_acc,
                        target_acc = target_acc, 
                        est=args.est)


