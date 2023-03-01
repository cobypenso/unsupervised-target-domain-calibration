#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import csv
import pre_process as prep
from torch.utils.data import DataLoader
from data_list import ImageList, MultiTransImageList
import multiprocessing
import sys
pytorch_path = os.path.abspath(os.path.dirname(__file__))
project_path = os.path.dirname(pytorch_path)
MDD_path = os.path.join(project_path,'MDD')
sys.path.extend([project_path])
sys.path.extend([MDD_path])


def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def generate_feature_wrapper(loader, model,dir,output_name=None):

    def gather_outputs(selected_loader, output_name):
        if 'MDD' in output_name:
            base_network, bottleneck_layer, classifier_layer = model[0], model[1], model[2]
            
        elif 'MCD' in output_name:
            feature_extractor, C1, C2 = model[0],model[1], model[2]
            
        with torch.no_grad():
            start_test = True
            iter_loader = iter(selected_loader)
            for i in range(len(selected_loader)):
                inputs, labels = iter_loader.next()
                inputs = inputs.cuda()
                if 'MDD' in output_name:
                    conv_features = base_network(inputs)
                    fc_features = bottleneck_layer(conv_features)
                    logit = classifier_layer(fc_features)

                elif 'MCD' in output_name:
                    fc_features = feature_extractor(inputs)
                    output_C1 = C1(fc_features)
                    output_C2 = C2(fc_features)
                    logit = (output_C1 + output_C2) / 2.0

                else:
                    fc_features, logit = model(inputs)

                if start_test:
                    features_ = fc_features.float().cpu()
                    outputs_ = logit.float().cpu()
                    labels_ = labels
                    start_test = False
                else:
                    features_ = torch.cat((features_, fc_features.float().cpu()), 0)
                    outputs_ = torch.cat((outputs_, logit.float().cpu()), 0)
                    labels_ = torch.cat((labels_, labels), 0)
            return features_, outputs_, labels_


    def save(loader, output_name, data_name):
        features, outputs, labels = gather_outputs(loader, output_name)
        np.save(dir + '/' + output_name + '_' + data_name + '_feature.npy', features)
        np.save(dir + '/' + output_name + '_' + data_name + '_output.npy', outputs)
        np.save(dir + '/' + output_name + '_' + data_name + '_label.npy', labels)


    print("-----------------Saving:target-----------")
    save(loader['target'], output_name, 'target')
    print("-----------------Saving:source_train-----------")
    save(loader['source_train'], output_name, 'source_train')
    print("-----------------Saving:source_val-----------")
    save(loader['source_val'], output_name, 'source_val')
    

def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    prep_dict["test_10crop"] = prep.image_test_10crop(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source_train"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    # ------------------------------------------------------------
    # ------------------------ TARGET DATA -----------------------

    dsets["target"] = MultiTransImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=[prep_dict["test"], prep_dict["target"]], val_ratio=config['target_val_ratio'], ds_type='train', m = config['num_of_aug'])
    # Get also validation set of target (for TargetPseudoLabelCalibration - TPLC)
    dsets["target_val"] = MultiTransImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=[prep_dict["test"], prep_dict["target"]], val_ratio=config['target_val_ratio'], ds_type='test', m = config['num_of_aug'])

    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=False, num_workers=4)

    dset_loaders["target_val"] = DataLoader(dsets["target_val"], batch_size=train_bs, \
                                        shuffle=False, num_workers=0)



    # ------------------------------------------------------------
    # ------------------------ SOURCE DATA -----------------------
    dsets["source_train"] = ImageList(open(data_config["source_train"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
    dset_loaders["source_train"] = DataLoader(dsets["source_train"], batch_size=train_bs, \
            shuffle=False, num_workers=4)
    dsets["source_val"] = ImageList(open(data_config["source_val"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
    dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=train_bs, \
                                        shuffle=False, num_workers=4)


    

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=4)


    # ------------------------------------------------------------
    # ------------------------ NETWORKS --------------------------

    backbone = torch.load(args.output_folder + '/snapshot/' + config["dataset"] + '/' + config['method'] + '/' + config['task'] + '/best_model.pth.tar')
    backbone = backbone.cuda()
    backbone.train(False)

    feature_dir = os.path.join(args.output_folder, 'features', config["dataset"], config['method'])
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    
    output_name = config["dataset"] + '_' + config['method'] + '_' + config['task']
    print(output_name)


    # ------------------------------------------------------------
    # ----------------- GENERATE FEATURES ------------------------

    generate_feature_wrapper(dset_loaders, backbone, feature_dir, output_name=output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default= 'CDAN+E', choices=['CDAN+E', 'DANN', 'DANN+E'])
    parser.add_argument('--net', type=str, default='ResNet18', choices=["ResNet18"])

    
    parser.add_argument('--dataset', type=str, default='office-home', choices=['office31', 'visda', 'office-home'],
                        help="The dataset or source dataset used")

    parser.add_argument('--target_val_ratio', type=float, default=0.2)                  
    parser.add_argument('--source', type=str, default='Real_World')
    parser.add_argument('--target', type=str, default='Art')
    parser.add_argument('--output_folder', type=str)
    # -----------------------------------


    # name of dataset - source
    parser.add_argument('--dset', type=str, default='office-home', choices=['office31', 'visda', 'office-home'],
                        help="The dataset or source dataset used")    # source dataset path
    parser.add_argument('--s_train_dset_path', type=str, default=None, help="The source train dataset path list")
    

    parser.add_argument('--s_val_dset_path', type=str,
                        default=None, help="The source validation dataset path list")
    parser.add_argument('--t_dset_path', type=str,
                        default=None, help="The target dataset path list")


    parser.add_argument('--num_of_aug_for_pl', type=int, default=1)
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    parser.add_argument('--use_backbone_aux', type=str, default='False', choices=['True', 'False'])
    
    args = parser.parse_args()
    args.s_train_dset_path = f'../data/{args.dataset}/{args.source}_train_list.txt'
    if args.dataset == 'domainnet':
        args.s_val_dset_path = '../data/' + args.dataset + '/' + args.source + '_test_list.txt'
        args.t_dset_path = '../data/' + args.dataset + '/' + args.target + '_test_list.txt'
    else:
        args.s_val_dset_path ='../data/' + args.dataset + '/' + args.source + '_val_list.txt'
        args.t_dset_path='../data/' + args.dataset + '/' + args.target + '_list.txt'




    # train config
    config = {}
    config['num_of_aug'] = args.num_of_aug_for_pl
    config['method'] = args.method
    config['task'] = os.path.basename(args.s_train_dset_path)[0].upper() + '2' + os.path.basename(args.t_dset_path)[0].upper()


    config['target_val_ratio'] = args.target_val_ratio

    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
    config["loss"] = {"trade_off":1.0}
    if "AlexNet" in args.net:
        config["prep"]['params']['alexnet'] = True
        config["prep"]['params']['crop_size'] = 227
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["loss"]["random"] = args.random
    config["loss"]["random_dim"] = 1024
    config["dataset"] = args.dset
    config["data"] = {"source_train":{"list_path":args.s_train_dset_path, "batch_size":36}, \
                        "source_val": {"list_path": args.s_val_dset_path, "batch_size": 36}, \
                        "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                        "test":{"list_path":args.t_dset_path, "batch_size":4}}

    if config["dataset"] == "office31":
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "image-clef":
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "visda":
        config["network"]["params"]["class_num"] = 12
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "domainnet":
        config["network"]["params"]["class_num"] = 345
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
    train(config)

   