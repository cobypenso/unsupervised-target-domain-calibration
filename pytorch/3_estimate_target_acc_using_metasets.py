#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os
import argparse
import pickle
import numpy as np
import torch
import network
import pre_process as prep
from torch.utils.data import DataLoader
from scipy import linalg
from tqdm import trange
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def calculate_activation_statistics(features):
    act = features
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


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
                    inputs_ = inputs.float().cpu()
                    outputs_ = logit.float().cpu()
                    features_ = fc_features.float().cpu()
                    labels_ = labels
                    start_test = False
                else:
                    inputs_ = torch.cat((inputs_, inputs.float().cpu()), 0)
                    features_ = torch.cat((features_, fc_features.float().cpu()), 0)
                    outputs_ = torch.cat((outputs_, logit.float().cpu()), 0)
                    labels_ = torch.cat((labels_, labels), 0)
            return inputs_, features_, outputs_, labels_


    def save(loader, output_name, data_name):
        inputs, features, outputs, labels = gather_outputs(loader, output_name)
        np.save(dir + '/' + output_name + '_' + data_name + '_inputs.npy', inputs)
        np.save(dir + '/' + output_name + '_' + data_name + '_output.npy', outputs)
        np.save(dir + '/' + output_name + '_' + data_name + '_label.npy', labels)
        np.save(dir + '/' + output_name + '_' + data_name + '_features.npy', features)

    print("-----------------Saving:source_val-----------")
    save(loader, output_name, 'source_val')
    

def full_pipeline(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    augs = prep.image_strong_aug(**config["prep"]['params'])
    for i in range(len(augs)):
        prep_dict[f"source_g{i}"] = augs[i]
    prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source_train"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    # ------------------------------------------------------------
    # ------------------------ SOURCE DATA -----------------------
    for i in range(len(augs)):
        dsets[f"source_val_{i}"] = ImageList(open(data_config["source_val"]["list_path"]).readlines(), \
                                    transform=prep_dict[f"source_g{i}"])
        dset_loaders[f"source_val_{i}"] = DataLoader(dsets[f"source_val_{i}"], batch_size=train_bs, \
                                            shuffle=False, num_workers=8)

                            

    original_source_val = ImageList(open(data_config["source_val"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])         
    dset_loaders["original_source_val"] = DataLoader(original_source_val, batch_size=train_bs, \
                                        shuffle=False, num_workers=8)                       

    # ------------------------------------------------------------
    # ------------------------ NETWORKS --------------------------

    backbone = torch.load(args.output_folder + '/snapshot/' + config["dataset"] + '/' + config['method'] + '/' + config['task'] + '/best_model.pth.tar')
    backbone = backbone.cuda()
    backbone.train(False)

    feature_dir = os.path.join(args.output_folder, 'metasets', config["dataset"], config['method'], config['task'])
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    
    output_name = 'meta_dataset'
    # ------------------------------------------------------------
    # ----------------- Augemtations  ----------------------------

    generate_feature_wrapper(dset_loaders['original_source_val'], backbone, feature_dir, output_name=output_name + '_original')
    compute_meta_dataset_accuracy_on_index(config=config, index = -1)
    m1, s1 = compute_original_dataset_FD(config)

    num_sets = config['num_sets']
    for i in trange(int(num_sets/len(augs))):
        for j in range(len(augs)):
            index = (i * len(augs) + j)
            generate_feature_wrapper(dset_loaders[f'source_val_{j}'], backbone, feature_dir, output_name=output_name + '_current')
            compute_meta_dataset_accuracy_on_index(config=config, index=index)
            compute_meta_dataset_FD_on_index(config=config, index=index, ref_stats = (m1, s1))

    return num_sets


def compute_meta_dataset_accuracy_on_index(config, index):
    num_sets = config['num_sets']
    feature_dir = os.path.join(args.output_folder, 'metasets', config["dataset"], config['method'], config['task'])
    output_name = 'meta_dataset'
    data_name = 'source_val'
    
    if index == -1:
        labels = np.load(feature_dir + '/' + output_name + '_original_' + data_name + '_label.npy')
        outputs = np.load(feature_dir + '/' + output_name + '_original_' + data_name + '_output.npy')
    else:
        labels = np.load(feature_dir + '/' + output_name + '_current_' + data_name + '_label.npy')
        outputs = np.load(feature_dir + '/' + output_name + '_current_' + data_name + '_output.npy')
        

    acc = (np.argmax(outputs, axis = -1) == labels).sum() / len(labels)
    correct = (np.argmax(outputs, axis = -1) == labels).sum()

    if index == -1:
        np.save(feature_dir + '/original_acc.npy', [acc, correct])
    else:
        np.save(feature_dir + f'/{index}_acc.npy', [acc, correct])


def compute_original_dataset_FD(config):
    feature_dir = os.path.join(args.output_folder, 'metasets', config["dataset"], config['method'], config['task'])
    output_name = 'meta_dataset'
    data_name = 'source_val'
    features1 = np.load(feature_dir + '/' + output_name + '_original_' + data_name + '_features.npy')
    m1, s1, act1 = calculate_activation_statistics(features1)
    # saving features for nn regression
    np.save(feature_dir + '/' + f'original_variance', s1)
    np.save(feature_dir + '/' +f'original_mean', m1)

    return m1, s1

def compute_meta_dataset_FD_on_index(config, index, ref_stats = None):
    num_sets = config['num_sets']
    feature_dir = os.path.join(args.output_folder, 'metasets', config["dataset"], config['method'], config['task'])
    output_name = 'meta_dataset'
    data_name = 'source_val'

    if type(ref_stats) == type(None):
        features1 = np.load(feature_dir + '/' + output_name + '_original_' + data_name + '_features.npy')
        m1, s1, act1 = calculate_activation_statistics(features1)
    else:
        m1, s1 = ref_stats


    features2 = np.load(feature_dir + '/' + output_name + f'_current_' + data_name + '_features.npy')
    m2, s2, act2 = calculate_activation_statistics(features2)
    fd_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('FD: ', fd_value)

    # saving features for nn regression
    np.save(feature_dir + '/' + f'{index}_variance', s2)
    np.save(feature_dir + '/' +f'{index}_mean', m2)
    np.save(feature_dir + '/' +f'{index}_fd', fd_value)
    


def train_linear_regressor_step(num_sets):
    # data preparation
    feature_dir = os.path.join(args.output_folder, 'metasets', config["dataset"], config['method'], config['task'])
    data_name = 'source_val'
    
    variance_shape = np.load(feature_dir + "/1_variance.npy").shape
    mean_shape = np.load(feature_dir + "/1_mean.npy").shape

    fd_list = np.zeros(num_sets)
    acc_list = np.zeros(num_sets)
    var_list = np.zeros(shape = (num_sets, *variance_shape))
    mean_list = np.zeros(shape = (num_sets, *mean_shape))
    for idx in range(num_sets):
        fd_list[idx] = np.load(feature_dir + f"/{idx}_fd.npy")
        var_list[idx] = np.load(feature_dir + f"/{idx}_variance.npy")
        mean_list[idx] = np.load(feature_dir + f"/{idx}_mean.npy")
        acc_list[idx] = np.load(feature_dir + f"/{idx}_acc.npy")[0]


    # Choose some sample sets as validation (also used in NN regression)
    indice = 5
    train_data = fd_list[indice:]
    train_acc = acc_list[indice:]
    test_data = fd_list[:indice]
    test_acc = acc_list[:indice]
    # linear regression
    slr = LinearRegression()
    slr.fit(train_data.reshape(-1, 1), train_acc.reshape(-1, 1))
    test_pred = slr.predict(test_data.reshape(-1, 1))

    # evaluation with metrics
    print('Test on Validation Set..')
    R2 = r2_score(test_acc, slr.predict(test_data.reshape(-1, 1)))
    RMSE = mean_squared_error(test_acc, slr.predict(test_data.reshape(-1, 1)), squared=False)
    MAE = mean_absolute_error(test_acc, slr.predict(test_data.reshape(-1, 1)))
    print('\nTest set: R2 :{:.4f} RMSE: {:.4f} MAE: {:.4f}\n'.format(R2, RMSE, MAE))

    # analyze the statistical correlation
    rho, pval = stats.spearmanr(test_data, test_acc)
    print('\nRank correlation-rho', rho)
    print('Rank correlation-pval', pval)

    rho, pval = stats.pearsonr(test_data, test_acc)
    print('\nPearsons correlation-rho', rho)
    print('Pearsons correlation-pval', pval)

    pickle.dump(slr, open(feature_dir + '/model.pkl', 'wb'))
    return slr




############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


def estimate_target_acc(config, model):
    feature_path = os.path.join(args.output_folder, 'features', config["dataset"], config['method'], config["dataset"] + '_' + config['method'] + '_' + config['task'])
    type = 'target'
    target_features = np.load(feature_path + f'_{type}_feature.npy')
    

    feature_dir = os.path.join(args.output_folder, 'metasets', config["dataset"], config['method'], config['task'])
    output_name = 'meta_dataset'
    data_name = 'source_val'
    # source_features = np.load(feature_dir + '/' + output_name + 'original_' + data_name + '_features.npy')
    # m1, s1, act1 = calculate_activation_statistics(source_features)
    m1 = np.load(feature_dir + "/original_mean.npy")
    s1 = np.load(feature_dir + "/original_variance.npy")

    m2, s2, act2 = calculate_activation_statistics(target_features)
    fd_value = calculate_frechet_distance(m1, s1, m2, s2)

    if model == None:
        model = pickle.load(open(feature_dir + '/model.pkl', 'rb'))

    acc = model.predict(fd_value.reshape(-1, 1))
    print (f'Estimated Acc: {acc}')
    return acc


############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default= 'CDAN+E', choices=['CDAN+E', 'DANN', 'DANN+E'])
    parser.add_argument('--net', type=str, default='ResNet18', choices=["ResNet18"])

    
    parser.add_argument('--dataset', type=str, default='office-home', choices=['office31', 'office-home'],
                        help="The dataset or source dataset used")


    parser.add_argument('--source', type=str, default='Art')
    parser.add_argument('--target', type=str, default='Real_World')
    parser.add_argument('--output_folder', type=str)
    # -----------------------------------


    # name of dataset - source
    parser.add_argument('--dset', type=str, default='office-home', choices=['office31', 'office-home'],
                        help="The dataset or source dataset used")    # source dataset path
    parser.add_argument('--s_train_dset_path', type=str, default=None, help="The source train dataset path list")
    

    parser.add_argument('--s_val_dset_path', type=str,
                        default=None, help="The source validation dataset path list")

    parser.add_argument('--new_version', type=bool, default=False)
    parser.add_argument('--num_sets', type=int, default=50)
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    
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
    config['method'] = args.method
    config['task'] = os.path.basename(args.s_train_dset_path)[0].upper() + '2' + os.path.basename(args.t_dset_path)[0].upper()



    config["prep"] = {"test_10crop":False, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
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


    config["dataset"] = args.dset
    config["data"] = {"source_train":{"list_path":args.s_train_dset_path, "batch_size":256}, \
                        "source_val": {"list_path": args.s_val_dset_path, "batch_size": 256}, \
                        "target":{"list_path":args.t_dset_path, "batch_size":256}, \
                        "test":{"list_path":args.t_dset_path, "batch_size":256}}

    if config["dataset"] == "office31":
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('Dataset cannot be recognized. Please define your own dataset here.')
 
    config['num_sets'] = args.num_sets


    full_pipeline(config=config)
    model = train_linear_regressor_step(config['num_sets'])
    acc = estimate_target_acc(config, model = model)
    task = config['task']
    np.save(f'accuracy_estimation/{args.dset}_{args.method}_{task}_est_acc.npy', acc)
