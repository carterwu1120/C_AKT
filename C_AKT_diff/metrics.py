import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels.reshape(-1)).double()
    correct = correct.sum()
    return correct / len(labels)

def compute_auc(all_target, all_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)

def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)

# Consider past predictions
def AUC_curve_1(save_dir, id, y_true, yt, questions):

    preds = np.zeros((len(y_true), len(y_true)))

    yt = yt[0]
    questions = questions[0][1:len(y_true) + 1]

    for num in range(len(y_true)):
        if num >= 1:
            preds[num][:num] = preds[num - 1][:num]
        for ques in range(num, len(y_true)):
            preds[num][ques] = yt[num][questions[ques]]

    auc_array = np.zeros(len(y_true))
    for num in range(len(y_true)):
        # try:
        #     auc = metrics.roc_auc_score(y_true, preds[num])
        # except ValueError:
        #     pass
        auc = metrics.roc_auc_score(y_true, preds[num])
        auc_array[num] = auc
    
    print('Response num, True, Current AUC: ')
    for num in range(len(y_true)):
        print('Response {:03d}: {}, {:.3f}'.format(num + 1, y_true[num], auc_array[num]))
    print()

    response_nums = [i + 1 for i in range(len(y_true))]
    plt.plot(response_nums, auc_array, 'r')
    plt.title('Student ID: ' + str(id))
    plt.xlabel('response')
    plt.ylabel('AUC')
    plt.ylim(0.4, 1)

    plt.savefig(os.path.join(save_dir, 'AUC_curve_1.png'))
    plt.cla()
    plt.clf()


# Mask the past predictions
def AUC_curve_2(save_dir, id, y_true, yt, questions):

    preds = np.zeros((len(y_true), len(y_true)))

    yt = yt[0]
    questions = questions[0][1:len(y_true) + 1]

    for num in range(len(y_true)):
        if num >= 1:
            preds[num][:num] = preds[num - 1][:num]
        for ques in range(num, len(y_true)):
            preds[num][ques] = yt[num][questions[ques]]

    auc_array = np.array([])
    for num in range(len(y_true)):
        sum = y_true[num:].sum()
        if (sum != len(y_true[num:])) and (sum != 0):
            auc = metrics.roc_auc_score(y_true[num:], preds[num][num:])
            auc_array = np.append(auc_array, auc)
        else:
            break

    response_nums = [i + 1 for i in range(len(auc_array))]
    plt.plot(response_nums, auc_array, 'r')
    plt.title('Student ID: ' + str(id))
    plt.xlabel('response')
    plt.ylabel('AUC')
    plt.ylim(0.4, 1)

    plt.savefig(os.path.join(save_dir, 'AUC_curve_2.png'))
    plt.cla()
    plt.clf()


def knowledge_mastery_baseline(save_dir, id, y_true, yt, questions):

    preds = np.zeros((len(y_true), len(y_true)))

    if len(yt.shape) == 3:
        yt = yt[0]

    if len(questions.shape) == 2:
        questions = questions[0][1:len(y_true) + 1]

    for num in range(len(y_true)):
        if num >= 1:
            preds[num][:num] = preds[num - 1][:num]
        for ques in range(num, len(y_true)):
            preds[num][ques] = yt[num][questions[ques]]
    
    best_thres = np.zeros(len(preds))

    for res in range(len(preds)):

        thres_choices = np.linspace(0, 1, 100)
        accuracy_array = np.zeros(len(thres_choices))

        preds_copy = np.zeros((len(thres_choices), len(preds[0])))
        for num_1 in range(len(thres_choices)):
            preds_copy[num_1] = preds[res]

        for num_2 in range(len(thres_choices)):

            preds_copy[num_2][preds_copy[num_2] > thres_choices[num_2]] = 1
            preds_copy[num_2][preds_copy[num_2] <= thres_choices[num_2]] = 0
            
            confusion = metrics.confusion_matrix(y_true, preds_copy[num_2])
            TP = confusion[1, 1]
            TN = confusion[0, 0]
            FP = confusion[0, 1]
            FN = confusion[1, 0]
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            accuracy_array[num_2] = accuracy
        
        temp = np.where(accuracy_array == np.max(accuracy_array))
        index = temp[0][0]
        thres = thres_choices[index]
        best_thres[res] = thres
    
    print('--- DKT ---')
    print('Max Accuracy: {:04f}'.format(np.max(accuracy_array)))
    print('Best Threshold: {:04f}'.format(np.max(best_thres)))
    print()

    box = np.ones(10)/10
    accuracy_array_smooth = np.convolve(accuracy_array, box, mode = 'same')

    plt.plot(thres_choices, accuracy_array_smooth, 'r')
    plt.title('Student ID: ' + str(id))
    plt.xlabel('threshold')
    plt.ylabel('accuracy')
    plt.ylim(0.4, 1)

    plt.savefig(os.path.join(save_dir, 'DKT_last_state_KMB.png'))
    plt.cla()
    plt.clf()

    response_nums = [i + 1 for i in range(len(y_true))]
    plt.plot(response_nums, best_thres, 'r')
    plt.title('Student ID: ' + str(id))
    plt.xlabel('response (time)')
    plt.ylabel('KMB')
    plt.ylim(0, 1)

    plt.savefig(os.path.join(save_dir, 'KMB_growth_curve.png'))
    plt.cla()
    plt.clf()

    return np.max(accuracy_array), np.max(best_thres)