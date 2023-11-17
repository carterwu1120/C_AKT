# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import os
import numpy as np
import torch

import matplotlib.pyplot as plt

from sklearn import metrics
from metrics import compute_accuracy, compute_auc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
        (1.0 - target) * np.log(np.maximum(1e-10, 1.0-pred))
    if mod == 'avg':
        return np.average(loss)*(-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False

def train_akt(net, optimizer, train_loader, label):
    net.train()

    pred_list = []
    target_list = []

    true_el = 0
    for idx, (features_problem, features_concept, questions_problem, questions_concept, diffculty_problem, diffculty_concept, answers) in enumerate(train_loader):
        optimizer.zero_grad()

        #target_1 = np.floor(answers)
        #el = np.sum(target_1 >= -.9)
        #element_count += el

        
        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()
        input_problem_diff = diffculty_problem.float()
        input_concept_diff  = diffculty_concept.float()
        target = answers.float()
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        input_diff = torch.cat((input_problem_diff.unsqueeze(0), input_concept_diff.unsqueeze(0)), 0)
        
        loss, pred, true_ct = net(input_q, input_qa, input_diff, target)
        pred = pred.detach().cpu().numpy()  # (seqlen * batch_size, 1)
        loss.backward()
        true_el += true_ct.cpu().numpy()

        optimizer.step()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = answers.reshape((-1,)).cpu().numpy()

        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc

def test_akt(net, optimizer, valid_loader, label):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    net.eval()
    
    pred_list = []
    target_list = []

    true_el = 0
    #element_count = 0
    for idx, (features_problem, features_concept, questions_problem, questions_concept, diffculty_problem, diffculty_concept, answers) in enumerate(valid_loader):

        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()
        input_problem_diff = diffculty_problem.float()
        input_concept_diff = diffculty_concept.float()
        target = answers.float()
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        input_diff = torch.cat((input_problem_diff.unsqueeze(0), input_concept_diff.unsqueeze(0)), 0)
        
        with torch.no_grad():
            #if pid_flag:
            #    loss, pred, ct = net(input_q, input_qa, target, input_pid)
            #else:
            loss, pred, ct = net(input_q, input_qa, input_diff, target)
        pred = pred.cpu().numpy()  # (seqlen * batch_size, 1)
        true_el += ct.cpu().numpy()
        #target = target.cpu().numpy()

        # correct: 1.0; wrong 0.0; padding -1.0
        target = answers.reshape((-1,)).cpu().numpy()
        nopadding_index = np.flatnonzero(target >= -0.9)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        #element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)
    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    #print(all_target)
    #print(all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc

def test_akt_one_student(model, test_loader, save_dir, id, st_len):

    model.eval()

    for idx, (features_problem, features_concept, questions_problem, questions_concept, diffculty_problem, diffculty_concept, answers) in enumerate(test_loader):

        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()
        input_problem_diff = diffculty_problem.float()
        input_concept_diff = diffculty_concept.float()
        target = answers.float()
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        input_diff = torch.cat((input_problem_diff.unsqueeze(0), input_concept_diff.unsqueeze(0)), 0)

        with torch.no_grad():
            loss, pred, ct = model(input_q, input_qa, input_diff, target, pid_data=None) # ct: Response num
        
        target = answers.reshape((-1,)).cpu().numpy()
        nopadding_index = np.flatnonzero(target >= -0.9)  # 原本是target >=-0.9， 但因為target裡的東西跟answer一樣（0 or 1）沒回答應該是-1所以只要有回答的都符合條件
        # nopadding_index = np.flatnonzero(target >= 0)    # Return indices that are non-zero in the flattened version of a.
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        pred_nopadding = pred_nopadding.cpu().numpy()
        pred_nopadding = pred_nopadding[:st_len]
        print(pred_nopadding)
        target_nopadding = target[nopadding_index]
        target_nopadding = target_nopadding[:st_len]
        print(target_nopadding)
        # print("--- AKT ---")

        # print("Auccracy: {:04f}".format(compute_accuracy(target_nopadding, pred_nopadding)))
        print("AUC: {:04f}".format(compute_auc(target_nopadding, pred_nopadding)))

        # thres_choices = np.linspace(0, 1, 100)  # 0~1 間距 100
        # AUC_array = np.zeros(len(thres_choices))    # Return a new array of given shape and type, filled with zeros. e.g., (2,3) or 2
        # preds_copy = np.zeros((len(thres_choices), len(pred_nopadding)))

        # # accuracy
        # accuracy_array = []
        # pi_V = pred_nopadding.copy()
        # for i in thres_choices:
        #     pi_V_temp = np.zeros(len(pi_V))
        #     pi_V_temp += pi_V
        #     for pi, pm in enumerate(pi_V_temp):
        #         if pm > i:
        #             pi_V_temp[pi] = 1
        #         else:
        #             pi_V_temp[pi] = 0
        #     accuracy_array.append(metrics.accuracy_score(target_nopadding, pi_V_temp))
        

        # for num in range(len(thres_choices)):
        #     preds_copy[num] = pred_nopadding.copy()

        # for num in range(len(thres_choices)):
        #     preds_copy[num][preds_copy[num] > thres_choices[num]] = 1
        #     preds_copy[num][preds_copy[num] <= thres_choices[num]] = 0
                      
        #     confusion = metrics.confusion_matrix(target_nopadding, preds_copy[num])
        #     # print(confusion)
        #     TP = confusion[1, 1]
        #     TN = confusion[0, 0]
        #     FP = confusion[0, 1]
        #     FN = confusion[1, 0]
        #     AUC = (TP + TN) / (TP + TN + FP + FN)
        #     AUC_array[num] = AUC
        
        # temp = np.where(AUC_array == np.max(AUC_array))
        
        # index = temp[0][0]
        # best_thres = thres_choices[index]

        # print('--- AKT ---')
        # print("Accuracy: {:04f}".format(np.max(accuracy_array)))
        # print('Max AUC: {:04f}'.format(np.max(AUC_array)))
        # print('Best Threshold: {:04f}'.format(best_thres))
        # print()

        # # box = np.ones(10)/10
        # # AUC_array_smooth = np.convolve(AUC_array, box, mode = 'same')

        # plt.plot(thres_choices, AUC_array, 'r')
        # plt.title('Student ID: ' + str(id))
        # plt.xlabel('threshold')
        # plt.ylabel('AUC')
        # plt.ylim(0.4, 1)

        # plt.savefig(os.path.join(save_dir, 'AKT_last_state_KMB.png'))
        # plt.cla()
        # plt.clf()
