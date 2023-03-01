import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
from torch.utils.data import DataLoader
import lr_schedule
from data_list import ImageList

import wandb

def image_classification_test(loader, model, test_10crop=True):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(10)]
            for i in range(len(loader['test'][0])):
                data = [next(iter_test[j]) for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                for j in range(10):
                    inputs[j] = inputs[j].cuda()
                labels = labels
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
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
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


def train(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, \
            shuffle=True, num_workers=0, drop_last=True)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"], val_ratio=config['target_val_ratio'], ds_type='train')
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
            shuffle=True, num_workers=0, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"][i]) for i in range(10)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                shuffle=False, num_workers=0) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                shuffle=False, num_workers=0)

    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        if 'DANN' in config['method']:
            ad_net = network.AdversarialNetwork(base_network.output_num(), 1024)
        else:
            ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    best_model = None
    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == config["test_interval"] - 1:
            base_network.train(False)
            temp_acc = image_classification_test(dset_loaders, \
                base_network, test_10crop=prep_config["test_10crop"])
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str+"\n")
            config["out_file"].flush()
            print(log_str)

            wandb.log({'iter': i,
                       'precision': temp_acc})

        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network), osp.join(config["output_path"], \
                "iter_{:05d}_model.pth.tar".format(i)))
            if best_model != None:
                torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))

        loss_params = config["loss"]                  
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = next(iter_source)
        inputs_target, labels_target = next(iter_target)
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(i), random_layer)
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        elif config['method'] == 'DANN+E':
            transfer_loss = loss.DANN(features, ad_net)

        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        if config['method'] == 'DANN+E':
            entropy_loss = loss.EntropyLoss(nn.Softmax(dim=1)(outputs))
            total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss + 0.1 * entropy_loss
        else:
            total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        if i % 10 == 0:
            log_str = "iter: {:05d}, classifier_loss: {:.5f}".format(i, classifier_loss)
            print(log_str)
            wandb.log({'iter': i,
                       'classifier_loss': classifier_loss,
                       'total_loss': total_loss})

        total_loss.backward()
        optimizer.step()
        
    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--method', type=str, default='DANN', choices=['CDAN+E', 'DANN', 'DANN+E'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default=3, help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet18')

    parser.add_argument('--dset', type=str, default='office-home', choices=['office31', 'visda', 'office-home'],
                        help="The dataset or source dataset used")

    parser.add_argument('--target_val_ratio', type=float, default=0.2)
    parser.add_argument('--source', type=str, default='Real_World')
    parser.add_argument('--target', type=str, default='Art')

    parser.add_argument('--output_folder', type=str, default='.')

    parser.add_argument('--max_iterations', type=int, default=50000, help="")
    parser.add_argument('--test_interval', type=int, default=1000, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--random', type=bool, default=False, help="whether use random projection")
    args = parser.parse_args()

    if args.dset == 'domainnet':
        args.target_train_path = '../data/' + args.dset + '/' + args.target + '_train_list.txt'
        args.target_test_path = '../data/' + args.dset + '/' + args.target + '_test_list.txt'
    else:
        args.target_train_path = '../data/' + args.dset + '/' + args.target + '_list.txt'
        args.target_test_path = '../data/' + args.dset + '/' + args.target + '_list.txt'

    args.source_train_path = '../data/' + args.dset + '/' + args.source + '_train_list.txt'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # train config
    config = {}
    config['method'] = args.method
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.max_iterations
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.output_folder + "/snapshot/" + args.dset + "/" + args.method + '/' + osp.basename(args.source_train_path)[0].upper() + '2' + osp.basename(args.target_train_path)[0].upper()
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])

    config['target_val_ratio'] = args.target_val_ratio
    config["dataset"] = args.dset
    config["data"] = {"source":{"list_path":args.source_train_path, "batch_size":36}, \
                    "target":{"list_path":args.target_train_path, "batch_size":36}, \
                    "test":{"list_path":args.target_test_path, "batch_size":4}}


    config["prep"] = {"test_10crop":True, 'params':{"resize_size":256, "crop_size":224, 'alexnet':False}}
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

    config["optimizer"] = {"type":optim.SGD, "optim_params":{'lr':args.lr, "momentum":0.9, \
                        "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                        "lr_param":{"lr":args.lr, "gamma":0.001, "power":0.75} }

    # ----------- Dataset to train on ----------------------------------- #
    #  ------------------------------------------------------------------ #
    #  ------------------------------------------------------------------ #
    if config["dataset"] == "office31":
        config["network"]["params"]["class_num"] = 31
        if 'CDAN' in config['method']:
            if ("amazon" in args.source_train_path and "webcam" in args.target_train_path) or \
            ("webcam" in args.source_train_path and "dslr" in args.target_train_path) or \
            ("webcam" in args.source_train_path and "amazon" in args.target_train_path) or \
            ("dslr" in args.source_train_path and "amazon" in args.target_train_path):
                config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
            elif ("amazon" in args.source_train_path and "dslr" in args.target_train_path) or \
                ("dslr" in args.source_train_path and "webcam" in args.target_train_path):
                config["optimizer"]["lr_param"]["lr"] = 0.0003 # optimal parameters
    elif config["dataset"] == "visda":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 12
        config['loss']["trade_off"] = 1.0
    elif config["dataset"] == "office-home":
        config["optimizer"]["lr_param"]["lr"] = 0.001 # optimal parameters
        config["network"]["params"]["class_num"] = 65
    else:
        pass
    config["out_file"].write(str(config) + '\n')
    config["out_file"].flush()

    #  ------------------------------------------------------------------ #
    #  ------------------------------------------------------------------ #
    #  ------------------------------------------------------------------ #

    with wandb.init(project='uda'):
        wandb.config.update(config)
        train(config)