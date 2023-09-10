import torch
from torch import nn, optim
from torch.nn import functional as F
from scipy import optimize
from scipy.optimize import minimize
import numpy as np
import csv
import seaborn as sns
from torch.nn.parameter import Parameter
from sklearn import linear_model



def get_weight_union(source_train_feature, target_feature, source_val_feature):
    """
    :param source_train_feature: shape [n_tr, d], features from training set
    :param target_feature: shape [n_t, d], features from test set
    :param source_val_feature: shape [n_v, d], features from validation set

    :return:
    """
    print("-"*30 + "get_weight" + '-'*30)
    n_tr, d = source_train_feature.shape
    n_t, _d = target_feature.shape
    n_v, _d = source_val_feature.shape
    print("n_tr: ", n_tr, "n_v: ", n_v, "n_t: ", n_t, "d: ", d)

    if n_tr < n_t:
        sample_index = np.random.choice(n_tr,  n_t, replace=True)
        source_train_feature = source_train_feature[sample_index]
        sample_num = n_t
    elif n_tr > n_t:
        sample_index = np.random.choice(n_t, n_tr, replace=True)
        target_feature = target_feature[sample_index]
        sample_num = n_tr

    combine_feature = np.concatenate((source_train_feature, target_feature))
    combine_label = np.asarray([1] * sample_num + [0] * sample_num, dtype=np.int32)
    domain_classifier = linear_model.LogisticRegression()
    domain_classifier.fit(combine_feature, combine_label)
    domain_out = domain_classifier.predict_proba(source_val_feature)
    weight = domain_out[:, :1] / domain_out[:, 1:]
    return weight

class TempScaling(nn.Module):
    def __init__(self):
        super(TempScaling, self).__init__()

    def find_best_T(self, logits, labels, optimizer = 'fmin'):
        nll_criterion = nn.CrossEntropyLoss(reduce=False)
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = torch.mean(nll_criterion(scaled_logits, labels))
            return loss

        if optimizer == 'fmin':
            optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
            self.temperature = optimal_parameter[0]

        elif optimizer == 'brute':
            rranges = (slice(1,10,0.05),)
            optimal_parameter = optimize.brute(eval, rranges, finish=optimize.fmin)
            self.temperature = optimal_parameter[0]

        # elif optimizer == 'brute':
        #     rranges = np.arange(0.9,4,0.01)
        #     best_ece = 100
        #     best_tmp = None
        #     for tmp in rranges:
        #         ece = eval(tmp)
        #         if ece < best_ece:
        #             best_ece = ece
        #             best_tmp = tmp

        #     self.temperature = best_tmp
            
        return self.temperature.item()

class TempScalingOnECE(nn.Module):
    def __init__(self):
        super(TempScalingOnECE, self).__init__()

    def find_best_T(self, logits, labels, optimizer = 'fmin'):
        ece_criterion = ECELoss()
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = ece_criterion(scaled_logits, labels)
            return loss


        if optimizer == 'fmin':
            optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
            self.temperature = optimal_parameter[0]

        elif optimizer == 'brute':
            rranges = (slice(1,10,0.05),)
            optimal_parameter = optimize.brute(eval, rranges, finish=optimize.fmin)
            self.temperature = optimal_parameter[0]
        # elif optimizer == 'brute':
        #     rranges = np.arange(0.9,4,0.01)
        #     best_ece = 100
        #     best_tmp = None
        #     for tmp in rranges:
        #         ece = eval(tmp)
        #         if ece < best_ece:
        #             best_ece = ece
        #             best_tmp = tmp

        #     self.temperature = best_tmp

        return self.temperature.item()

class TempScalingOnEceGivenAcc(nn.Module):
    def __init__(self, acc_fix = None):
        super(TempScalingOnEceGivenAcc, self).__init__()
        self.n_bins = 15
        self.acc_fix = acc_fix

    def find_best_T(self, logits, acc_list=None, source_logits=None, source_labels=None):
        ece_criterion = ECELoss(n_bins=self.n_bins)
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = ece_criterion.forward_given_acc(scaled_logits, acc_list)
            return loss


        acc_list = get_accuracy_of_bins(source_logits, source_labels, bins = 15, ada=False)
        rranges = np.arange(0.5,4,0.01)
        best_ece = 100
        best_tmp = None
        for tmp in rranges:
            if type(self.acc_fix) != type(None):
                acc_list = [item * self.acc_fix.item() for item in acc_list]
            ece = eval(tmp)
            if ece < best_ece:
                best_ece = ece
                best_tmp = tmp

        self.temperature = best_tmp

        return self.temperature.item()


class TempScalingOnAdaEceGivenAcc(nn.Module):
    def __init__(self, acc_fix = None):
        super(TempScalingOnAdaEceGivenAcc, self).__init__()
        self.n_bins = 15
        self.acc_fix = acc_fix

    def find_best_T(self, logits, acc_list=None, source_logits=None, source_labels=None):
        ece_criterion = AdaptiveECELoss(n_bins=self.n_bins)
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = ece_criterion.forward_given_acc(scaled_logits, acc_list)
            return loss

        rranges = np.arange(0.5,4,0.01)
        best_ece = 100
        best_tmp = None
        acc_list = get_accuracy_of_bins(source_logits, source_labels, bins = 15, ada=True)
        for tmp in rranges:
            if type(self.acc_fix) != type(None):
                acc_list = [item * self.acc_fix.item() for item in acc_list]
            ece = eval(tmp)
            if ece < best_ece:
                best_ece = ece
                best_tmp = tmp

        self.temperature = best_tmp

        return self.temperature.item()


class VectorScalingModel(nn.Module):
    def __init__(self, class_num=65):
        super(VectorScalingModel, self).__init__()
        self.W = Parameter(torch.ones(class_num))
        self.b = Parameter(torch.zeros(class_num))

    def forward(self, logits):
        out = logits * self.W + self.b
        return out

class MatrixScalingModel(nn.Module):
    def __init__(self, class_num=65):
        super(MatrixScalingModel, self).__init__()
        self.W = Parameter(torch.eye(class_num))
        self.b = Parameter(torch.zeros(class_num))

    def forward(self, logits):
        out = torch.matmul(logits, self.W) + self.b
        return out

def VectorOrMatrixScaling(logits, labels, outputs_target, labels_target, cal_method=None):
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    class_num = logits.shape[1]

    if cal_method == 'VectorScaling':
        cal_model = VectorScalingModel(class_num=class_num).cuda()
    elif cal_method == 'MatrixScaling':
        cal_model = MatrixScalingModel(class_num=class_num).cuda()
    optimizer = optim.SGD(cal_model.parameters(), lr=0.01, momentum=0.9)

    logits = logits.cuda().float()
    labels = labels.cuda().long()
    outputs_target = outputs_target.cuda().float()
    labels_target = labels_target.cuda().long()

    # Calculate NLL and ECE before vector scaling or matrix scaling
    before_calibration_nll = nll_criterion(outputs_target, labels_target).item()
    before_calibration_ece = ece_criterion(outputs_target, labels_target).item()

    max_iter = 5000 
    for _ in range(max_iter):
        optimizer.zero_grad()
        out = cal_model(logits)
        loss = nn.CrossEntropyLoss().cuda()(out, labels)
        loss.backward()
        optimizer.step()
    final_output = cal_model(outputs_target)

    # Calculate NLL and ECE after temperature scaling
    after_calibration_nll = nll_criterion(final_output, labels_target).item()
    after_calibration_ece = ece_criterion(final_output, labels_target).item()

    return after_calibration_ece



class CPCS(nn.Module):
    def __init__(self):
        super(CPCS, self).__init__()

    def find_best_T(self, logits, labels, weight=None, optimizer = 'fmin'):
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            softmaxes = F.softmax(scaled_logits, dim=1)

            ## Transform to onehot encoded labels
            labels_onehot = torch.FloatTensor(scaled_logits.shape[0], scaled_logits.shape[1])
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.long().view(len(labels), 1), 1)
            brier_score = torch.sum((softmaxes - labels_onehot) ** 2, dim=1,keepdim = True)
            loss = torch.mean(brier_score * weight)
            return loss

        if optimizer == 'fmin':
            optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
            self.temperature = optimal_parameter[0]
        elif optimizer == 'brute':
            rranges = np.arange(0.9,4,0.01)
            best_ece = 100
            best_tmp = None
            for tmp in rranges:
                ece = eval(tmp)
                if ece < best_ece:
                    best_ece = ece
                    best_tmp = tmp

            self.temperature = best_tmp

        return self.temperature


class TransCal(nn.Module):
    def __init__(self, bias_term=True, variance_term=True):
        super(TransCal, self).__init__()
        self.bias_term = bias_term
        self.variance_term = variance_term

    def find_best_T(self, logits, weight, error, source_confidence):
        def eval(x):
            "x ==> temperature T"

            scaled_logits = logits / x[0]

            "x[1] ==> learnable meta parameter \lambda"
            if self.bias_term:
                controled_weight = weight ** x[1]
            else:
                controled_weight = weight

            ## 1. confidence
            max_L = np.max(scaled_logits, axis=1, keepdims=True)
            exp_L = np.exp(scaled_logits - max_L)
            softmaxes = exp_L / np.sum(exp_L, axis=1, keepdims=True)
            confidences = np.max(softmaxes, axis=1)
            confidence = np.mean(confidences)
            ## 2. accuracy
            if self.variance_term:
                weighted_error = controled_weight * error
                cov_1 = np.cov(np.concatenate((weighted_error, controled_weight), axis=1), rowvar=False)[0][1]
                var_w = np.var(controled_weight, ddof=1)
                eta_1 = - cov_1 / (var_w)

                cv_weighted_error = weighted_error + eta_1 * (controled_weight - 1)
                correctness = 1 - error
                cov_2 = np.cov(np.concatenate((cv_weighted_error, correctness), axis=1), rowvar=False)[0][1]
                var_r = np.var(correctness, ddof=1)
                eta_2 = - cov_2 / (var_r)

                target_risk = np.mean(weighted_error) + eta_1 * np.mean(controled_weight) - eta_1 \
                              + eta_2 * np.mean(correctness) - eta_2 * source_confidence
                estimated_acc = 1.0 - target_risk
            else:
                weighted_error = controled_weight * error
                target_risk = np.mean(weighted_error)
                estimated_acc = 1.0 - target_risk

                ## 3. ECE on bin_size = 1 for optimizing.
            ## Note that: We still utilize a bin_size of 15 while evaluating,
            ## following the protocal of Guo et al. (On Calibration of Modern Neural Networks)
            loss = np.abs(confidence - estimated_acc)
            return loss

        # return best_tmp
        rranges = (slice(1,10,0.05), slice(0,1.01,0.01))
        optimal_parameter = optimize.brute(eval, rranges, finish=optimize.fmin)
        self.temperature = optimal_parameter[0]

        return self.temperature.item()

class Oracle(nn.Module):
    def __init__(self, ada = False):
        super(Oracle, self).__init__()
        self.ada = ada


    def find_best_T(self, logits, labels, optimizer = 'fmin'):
        if self.ada:
            ece_criterion = AdaptiveECELoss()
        else:
            ece_criterion = ECELoss()
        def eval(x):
            "x ==> temperature T"
            if type(x) != np.float64:
                x = torch.from_numpy(x)
            scaled_logits = logits.float() / x
            loss = ece_criterion(scaled_logits, labels)
            return loss

        if optimizer == 'fmin':
            optimal_parameter = optimize.fmin(eval, torch.Tensor([2.0]), disp=False)
            self.temperature = optimal_parameter[0]
        elif optimizer == 'brute':
            rranges = np.arange(0.9,4,0.01)
            best_ece = 100
            best_tmp = None
            for tmp in rranges:
                ece = eval(tmp)
                if ece < best_ece:
                    best_ece = ece
                    best_tmp = tmp

            self.temperature = best_tmp
        
        return self.temperature.item()



class AdaptiveECELoss(nn.Module):
    '''
    Compute Adaptive ECE
    '''
    def __init__(self, n_bins=15, LOGIT=True, small_ds = False):
        super(AdaptiveECELoss, self).__init__()
        self.nbins = n_bins
        self.LOGIT = LOGIT
        self.small_ds = small_ds

    def histedges_equalN(self, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, self.nbins + 1),
                     np.arange(npt),
                     np.sort(x))
    def forward(self, logits, labels):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(softmaxes, 1)
        else:
            confidences, predictions = torch.max(logits, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        #print(n,confidences,bin_boundaries)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    def forward_given_acc(self, logits, acc):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        confidences[confidences == 1] = 0.999999
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), self.histedges_equalN(confidences.cpu().detach()))
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        ece = torch.zeros(1, device=logits.device)

        for idx, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if (prop_in_bin.item() > 0):
                if acc[idx] is not None:
                    accuracy_in_bin = acc[idx]
                    accuracy_in_bin = min(accuracy_in_bin, 0.99)
                    accuracy_in_bin = max(accuracy_in_bin, 0.01)
                    avg_confidence_in_bin = confidences[in_bin].mean().float()

                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

class ECELoss(nn.Module):
    ##TODO: refer to temperature scaling code in the original paper
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    """
    def __init__(self, n_bins=15, LOGIT = True, small_ds = False):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.LOGIT = LOGIT
        self.small_ds = small_ds

    def forward(self, logits, labels, plot = False, ax = None, title = None):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        correctness = predictions.eq(labels)
        confidences[confidences == 1] = 0.999999
        ece = torch.zeros(1, device=logits.device)

        confidence_in_bins = []
        accuracy_in_bins = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if (prop_in_bin.item() > 0) and (in_bin.sum() > 10):
                accuracy_in_bin = correctness[in_bin].float().mean()
                accuracy_in_bin = min(accuracy_in_bin, 0.99)
                accuracy_in_bin = max(accuracy_in_bin, 0.01)
                avg_confidence_in_bin = confidences[in_bin].mean().float()

                confidence_in_bins.append(avg_confidence_in_bin)
                accuracy_in_bins.append(accuracy_in_bin)
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            else:
                confidence_in_bins.append(0)
                accuracy_in_bins.append(0)

        if ece < 0:
            raise Exception(f'ECE can not be negative - conf : {confidence_in_bins} acc: {accuracy_in_bins}')

        if plot:
            x = (self.bin_lowers + self.bin_uppers) / 2
            y = accuracy_in_bins
            ax.bar(x=x, height=x, width=self.bin_uppers[0], align='center', color='r', edgecolor='black', linewidth=1)
            ax.bar(x=x,height=y, width=self.bin_uppers[0], align='center', edgecolor='black', linewidth=1)
            ax.plot([0,1], [0,1], linestyle = "--", color="gray")
            ax.set_xlabel('Condifence', fontsize=22)
            ax.set_ylabel('Accuracy', fontsize=22)
            ax.set_title(title + f' ECE={ece.item():.2f}', fontsize=16)
            return ece, ax 

        return ece

    def forward_given_acc(self, logits, acc):
        if self.LOGIT:
            softmaxes = F.softmax(logits, dim=1)
        else:
            softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        confidences[confidences == 1] = 0.999999
        ece = torch.zeros(1, device=logits.device)

        for idx, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if (prop_in_bin.item() > 0):
                if acc[idx] is not None:
                    accuracy_in_bin = acc[idx]
                    accuracy_in_bin = min(accuracy_in_bin, 0.99)
                    accuracy_in_bin = max(accuracy_in_bin, 0.01)
                    avg_confidence_in_bin = confidences[in_bin].mean().float()

                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

def get_accuracy_of_bins(logits, labels, bins, ada = False):
    def histedges_equalN(nbins, x):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbins + 1),
                     np.arange(npt),
                     np.sort(x))


    accuracy_list = []
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    correctness = predictions.eq(labels)

    if ada:
        n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(bins, confidences.cpu().detach()))
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
    else:
        bin_boundaries = torch.linspace(0, 1, bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
 
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            accuracy_list.append(accuracy_in_bin)
        else:
            accuracy_list.append(0) 
    
    return accuracy_list

def get_confidence_of_bins(logits, bins, weight = None):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    confidence_list = []
    prop_in_bin_list = []
    if weight is not None:
        weight_list = []

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            avg_confidence_in_bin = confidences[in_bin].mean().float()
            confidence_list.append(avg_confidence_in_bin)
            if weight is not None:
                weight_list.append(weight[in_bin].sum())
        else:
            confidence_list.append(0)
            if weight is not None:
                weight_list.append(0)
        
        prop_in_bin_list.append(prop_in_bin.item())

    if weight is not None:
        return confidence_list, prop_in_bin_list, weight_list
    return confidence_list, prop_in_bin_list