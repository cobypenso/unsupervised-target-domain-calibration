import numpy as np
import torch
import torch.nn as nn
import csv
import os
import time
from utils import (
    ECELoss,
    CPCS,
    TransCal,
    Oracle,
    TempScalingOnECE,
    TempScalingOnEceGivenAcc,
    TempScalingOnAdaEceGivenAcc
)
import pandas as pd



def cal_acc_error(logit, label):
    softmaxes = nn.Softmax(dim=1)(logit)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(label)
    accuracy = accuracies.float().mean()
    confidence = confidences.float().mean()
    error = 1 - accuracies.float()
    error = error.view(len(error), 1).float().numpy()
    return accuracy, confidence, error



def sweep_on_calibration_methods(logits_source_val,
                                labels_source_val,
                                logits_target,
                                labels_target,
                                weight=None, 
                                bias_term=True, 
                                variance_term=True,
                                optimizer = 'brute',
                                methods = None,
                                acc_fix = None,
                                verbose=True):

    sweep_res = {}
    for cal_method in methods:
        res = calibration_in_DA(logits_source_val,
                      labels_source_val,
                      logits_target,
                      labels_target,
                      cal_method,
                      weight, 
                      bias_term, 
                      variance_term,
                      optimizer,
                      acc_fix,
                      verbose=verbose)
        sweep_res[res['cal_method']] = res

    return sweep_res


def calibration_in_DA(logits_source_val,
                      labels_source_val,
                      logits_target,
                      labels_target,
                      cal_method=None, weight=None, 
                      bias_term=True, variance_term=True,
                      optimizer = 'brute',
                      acc_fix = None,
                      verbose=True):

    ece_criterion = ECELoss()

    # ------------------------------- Baselines ----------------------------------------------- #

    if cal_method == 'TempScaling':
        cal_model = TempScalingOnECE()
        optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val, optimizer=optimizer)
    elif cal_method == 'CPCS':
        cal_model = CPCS()
        optimal_temp = cal_model.find_best_T(logits_source_val, labels_source_val, torch.from_numpy(weight), optimizer=optimizer)
    elif cal_method == 'TransCal':
        """calibrate the source model first and then attain the optimal temperature for the source dataset"""
        cal_model = TempScalingOnECE()
        optimal_temp_source = cal_model.find_best_T(logits_source_val, labels_source_val, optimizer=optimizer)
        _, source_confidence, error_source_val = cal_acc_error(logits_source_val / optimal_temp_source,
                                                                labels_source_val)

        cal_model = TransCal(bias_term, variance_term)
        optimal_temp = cal_model.find_best_T(logits_target.numpy(), weight, error_source_val,
                                                source_confidence.item())

    elif cal_method == 'Oracle':
        cal_model = Oracle()
        optimal_temp = cal_model.find_best_T(logits_target, labels_target, optimizer=optimizer)

    elif cal_method == 'Oracle_ada':
        cal_model = Oracle(ada=True)
        optimal_temp = cal_model.find_best_T(logits_target, labels_target, optimizer=optimizer)


    # ---------------- Unsupervised Target Domain Calibration (UTDC) --------------------------- #

    elif cal_method == 'UTDC':
        cal_model = TempScalingOnEceGivenAcc(acc_fix = acc_fix)
        optimal_temp = cal_model.find_best_T(logits_target, source_logits=logits_source_val, source_labels=labels_source_val, optimizer=optimizer)
    
    elif cal_method == 'UTDC_ada':
        cal_model = TempScalingOnAdaEceGivenAcc(acc_fix = acc_fix)
        optimal_temp = cal_model.find_best_T(logits_target, source_logits=logits_source_val, source_labels=labels_source_val, optimizer=optimizer)
    
    # ------------------------------------------------------------------------------------------ #

    else:
        print (f'{cal_method} not exists')


    after_cal_ece = ece_criterion(logits_target / optimal_temp, labels_target).item()

    if verbose:      
        print(cal_method, 'ECE after calib:', after_cal_ece, 'Optimal T:', optimal_temp)
         
    
    
    return {
        'cal_method': cal_method,
        'ece_on_target': after_cal_ece,
        'optimal_temp': optimal_temp
    }
