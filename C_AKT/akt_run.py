# Code reused from https://github.com/jennyzhang0215/DKVMN.git
import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
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
    for idx, (features_problem, features_concept, questions_problem, questions_concept, answers) in enumerate(train_loader):
        optimizer.zero_grad()

        #target_1 = np.floor(answers)
        #el = np.sum(target_1 >= -.9)
        #element_count += el

        
        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()

        target = answers.float()
        
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        
        loss, pred, true_ct = net(input_q, input_qa, target)
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
    for idx, (features_problem, features_concept, questions_problem, questions_concept, answers) in enumerate(valid_loader):

        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()
        target = answers.float()
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        with torch.no_grad():
            #if pid_flag:
            #    loss, pred, ct = net(input_q, input_qa, target, input_pid)
            #else:
            loss, pred, ct = net(input_q, input_qa, target)
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

    for idx, (features_problem, features_concept, questions_problem, questions_concept, answers) in enumerate(test_loader):

        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()
        target = answers.float()
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        with torch.no_grad():
            loss, pred, ct = model(input_q, input_qa, target, pid_data=None) # ct: Response num
        
        target = answers.reshape((-1,)).cpu().numpy()
        nopadding_index = np.flatnonzero(target >= -0.9)  # 原本是target >=-0.9， 但因為target裡的東西跟answer一樣（0 or 1）沒回答應該是-1所以只要有回答的都符合條件
        # nopadding_index = np.flatnonzero(target >= 0)    # Return indices that are non-zero in the flattened version of a.
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        pred_nopadding = pred_nopadding.cpu().numpy()
        pred_nopadding = pred_nopadding[:st_len]
        target_nopadding = target[nopadding_index]
        target_nopadding = target_nopadding[:st_len]
        print("--- AKT ---")

        print("AUC: {:04f}".format(compute_auc(target_nopadding, pred_nopadding)))
        print("Auccracy: {:04f}".format(compute_accuracy(target_nopadding, pred_nopadding)))

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
def test_akt_one_student_knowledge_state(model, test_loader, save_dir, id, st_len):
    model.eval()

    data_dir = "data"
    data_file = "junyi_100to999_dedup_PC.csv"
    path = os.path.join(data_dir, data_file)
    df_PC = pd.read_csv(path)
    df_PC['problem'], _ = pd.factorize(df_PC['problem_id'], sort=True)
    df_PC['concept'], _ = pd.factorize(df_PC['concept_id'], sort=True)
    df_PC['feature_problem'] = df_PC['problem']*2 + df_PC['correct']
    df_PC['feature_concept'] = df_PC['concept']*2 + df_PC['correct']

    pred_array = []

    for idx, (features_problem, features_concept, questions_problem, questions_concept, answers) in enumerate(test_loader):
        
        input_problem_q = questions_problem.long()
        input_concept_q = questions_concept.long()
        input_problem_qa = features_problem.long()
        input_concept_qa = features_concept.long()
        target = answers.float()
        input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
        input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
        with torch.no_grad():
            loss, pred, ct = model(input_q, input_qa, target, pid_data=None) # ct: Response num
        target = answers.reshape((-1,)).cpu().numpy()
        nopadding_index = np.flatnonzero(target >= -0.9)  # 原本是target >=-0.9， 但因為target裡的東西跟answer一樣（0 or 1）沒回答應該是-1所以只要有回答的都符合條件
        # nopadding_index = np.flatnonzero(target >= 0)    # Return indices that are non-zero in the flattened version of a.
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        pred_nopadding = pred_nopadding.cpu().numpy()
        pred_nopadding = pred_nopadding[:st_len]
        target_nopadding = target[nopadding_index]
        target_nopadding = target_nopadding[:st_len]
        AUC_array = []
        for i in range(st_len):
            try:
                AUC_array.append(compute_auc(target_nopadding[:i+1], pred_nopadding[:i+1]))
            except:
                pass
        
        box = np.ones(10)/10
        AUC_array_smooth = np.convolve(AUC_array, box, mode = 'same')
        x = np.linspace(0, len(AUC_array)-1, len(AUC_array))
        plt.plot(x, AUC_array, 'r', markersize=4)
        # plt.plot(x, AUC_array_smooth, 'r')
        plt.title('Student ID: ' + str(id))
        plt.xlabel('response')
        plt.ylabel('AUC')
        plt.savefig(os.path.join(save_dir, 'AUC Growth curve.png'))
        plt.cla()
        plt.clf()

        print("AUC: {:04f}".format(compute_auc(target_nopadding, pred_nopadding)))

        for _, row in df_PC.iterrows():
            input_questions_problem = torch.cat((questions_problem[0][:st_len], torch.tensor([row['problem']]), questions_problem[0][st_len:]), 0).unsqueeze(0)
            input_questions_concept = torch.cat((questions_concept[0][:st_len], torch.tensor([row['concept']]), questions_concept[0][st_len:]), 0).unsqueeze(0)
            input_features_problem = torch.cat((features_problem[0][:st_len], torch.tensor([row['feature_problem']]), features_problem[0][st_len:]), 0).unsqueeze(0)
            input_features_concept = torch.cat((features_concept[0][:st_len], torch.tensor([row['feature_concept']]), features_concept[0][st_len:]), 0).unsqueeze(0)
            input_answers = torch.cat((answers[0][:st_len], torch.tensor([1]), answers[0][st_len:]), 0).unsqueeze(0)
            input_problem_q = input_questions_problem.long()
            input_concept_q = input_questions_concept.long()
            input_problem_qa = input_features_problem.long()
            input_concept_qa = input_features_concept.long()
            target = input_answers.float()
            input_q = torch.cat((input_problem_q.unsqueeze(0), input_concept_q.unsqueeze(0)), 0)
            input_qa = torch.cat((input_problem_qa.unsqueeze(0), input_concept_qa.unsqueeze(0)), 0)
            with torch.no_grad():
                loss, pred, ct = model(input_q, input_qa, target, pid_data=None) # ct: Response num
            pred_array.append(pred[st_len])

        df_PC['concept_id'] = df_PC['concept'].astype(str) + '_' + df_PC['concept_id']
        concept_array = df_PC['concept'].unique().tolist()
        concept_id_array = df_PC['concept_id'].unique().tolist() 
        df_PC['preduction'] = pd.DataFrame(pred_array)
        
        total_pred_array = df_PC.groupby(['concept_id'])['preduction'].sum()
        count_concept = df_PC['concept_id'].value_counts()
        ave_pred_array = [total_pred_array[i]/count_concept[i] for i in concept_id_array]
        data = pd.DataFrame({'x':ave_pred_array, 'y':concept_array, 'labels':concept_id_array, 'predictions':ave_pred_array})
        for i, row in data.iterrows():
            plt.annotate(row['y'], (row['x'], row['y']), textcoords='offset points', xytext=(0, 6), ha='center')

        
        sns.scatterplot(data=data, x='knowledge state', y='concept', hue='labels', size='predictions')
        plt.legend(bbox_to_anchor=(1.01, 1), fontsize='x-small', loc='upper left', borderaxespad=0)
        plt.savefig(os.path.join(save_dir, 'concept_knowledge_state.png'))
        # Display the plot
        plt.show()