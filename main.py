from email.parser import Parser
import os
import random
import pickle
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import KTDataset, pad_collate_akt
from c_akt_model import AKT
from akt_run import train_akt, test_akt, test_akt_one_student, test_akt_one_student_knowledge_state

# from diffculty.dataset_diff_cembed import KTDataset, pad_collate_akt
# from diffculty.akt_model_diff_cembed import AKT
# from diffculty.akt_run_diff import train_akt, test_akt, test_akt_one_student

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# To train
# python main.py --mode=0 --model=AKT --load_dir=

# To test and calculate knowledfe state
# python main.py --load_dir=train_C_AKT_Junyi_100to499 --data_file=junyi_500to999_dedup.csv --data_pc_file=junyi_100to499_dedup_PC.csv --response_count_file=junyi_500to999_dedup_count.csv --student_i=0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = int, default = 0, help = 'mode 0 is train, mode 1 is test')
    parser.add_argument('--model', type = str, default = 'AKT', help = 'choose the used model: DKT, AKT')
    parser.add_argument('--data_dir', type = str, default = 'data_junyi', help = 'Data dir for loading input data.')
    parser.add_argument('--load_dir', type = str, default = '', help = 'Where to load the trained model if finetunning. ' + 'Leave empty to train from scratch')
    parser.add_argument('--save_dir', type = str, default = 'logs', help = 'Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--data_file', type = str, default = 'junyi_100to499_dedup.csv.csv', help = 'Name of input data file.')
    parser.add_argument('--data_pc_file', type = str, default = 'junyi_100to499_dedup_PC.csv', help = 'Number of problems and concepts in dataset.')
    parser.add_argument('--response_count_file', type = str, default = 'junyi_100to499_dedup_count.csv', help = 'Number of responses per student.')
    parser.add_argument('--start_point', type = int, default = 0, help = 'Start to calculate AUC in which response (start point).')
    #parser.add_argument('--end-point', type = int, default = 182, help = 'Stop to calculate AUC in which response (end point).')
    parser.add_argument('--student_i', type = int, default = 0, help = 'To test the number i student in dataset.')
    # parser.add_argument('--data_type', type = str, default = 'problem', help = 'Type of input data (problem or concept).')
    parser.add_argument('--seed', type = int, default = 123, help = 'Random seed.')
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'Initial learning rate, DKT:1e-3, AKT:1e-5')
    parser.add_argument('--epochs', type = int, default = 300, help = 'Number of epochs to train.')
    parser.add_argument('--lr_decay', type = int, default = 10, help = 'After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type = float, default = 0.9, help = 'LR decay factor.')
    parser.add_argument('--shuffle', type = bool, default = False, help = 'Whether to shuffle the dataset or not.')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'Number of samples per batch.')
    
    return parser

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #CPU
    torch.cuda.manual_seed(seed) #GPU
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False

def save_train(save_dir):
    log = None

    #exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')
    model_file_name = args.model
    save_dir = '{}/train_{}/'.format(save_dir, model_file_name + timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_file = os.path.join(save_dir, model_file_name + '.pt')
    meta_file = os.path.join(save_dir, 'metadata.pkl')
    log_file = os.path.join(save_dir, timestamp + '_log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, 'wb'))

    return log, save_dir, model_file

def save_test(save_dir):
    log = None

    #exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H-%M-%S')
    model_file_name = args.model
    save_dir = '{}/test_{}/'.format(save_dir, model_file_name + timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    meta_file = os.path.join(save_dir, 'metadata.pkl')
    log_file = os.path.join(save_dir, timestamp + '_log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, 'wb'))

    return log, save_dir

def load_model(load_dir, save_dir):
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S')

    save_dir = '{}/{}/'.format(save_dir, load_dir)

    model_file = os.path.join(save_dir, 'AKT.pt')    

    # store data by dictionary
    # dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'loss', 'auc'])
    checkpoint = torch.load(model_file, map_location = device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if args.mode == 0:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    #best_val_loss = checkpoint['loss']
    best_val_auc = checkpoint['auc']
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    meta_file = os.path.join(save_dir, 'metadata.pkl')
    log_file = os.path.join(save_dir, timestamp + '_log.txt')
    log = open(log_file, 'w')

    # metadata.pkl: {'args': Namespace(get_parser裡的東西)}
    pickle.dump({'args': args}, open(meta_file, 'wb'))

    return log, save_dir, model_file, start_epoch, best_val_auc

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    init_seed(args.seed)

    # return dim, data_loader, id, point
    kt_dataset = KTDataset(data_dir = args.data_dir, 
                    data_file = args.data_file, 
                    data_pc_file = args.data_pc_file, 
                    response_count_file = args.response_count_file, 
                    student_i = args.student_i, 
                    mode = args.mode)  
    
    train_ratio = 0.9
    val_ratio = 0.1
    
    stu_num = kt_dataset.get_stu_num()
    train_size = int(train_ratio * stu_num)
    valid_size = stu_num - train_size

    # bug on dataset size
    train_dataset, val_dataset = torch.utils.data.random_split(kt_dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 0, collate_fn = pad_collate_akt)
    valid_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 0, collate_fn = pad_collate_akt)
    test_loader = DataLoader(kt_dataset, batch_size = args.batch_size, shuffle = args.shuffle, num_workers = 0, collate_fn = pad_collate_akt)         
 
 
    n_question = 15349

    model = AKT(n_question=n_question, n_pid=-1, n_blocks=1, d_model=256,
                dropout=0.05, kq_same=1, model_type='akt', d_ff=1024, l2=1e-5).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = lr_scheduler.StepLR(optimizer, step_size = args.lr_decay, gamma = args.gamma)

    if args.load_dir:
        log, save_dir, model_file, start_epoch, best_valid_auc = load_model(args.load_dir, args.save_dir)
    else:
        log, save_dir, model_file = save_train(args.save_dir)
        start_epoch = 0  
        best_valid_auc = 0 

    if args.mode == 0:
        # ==== start training ====
        best_epoch = start_epoch
        for epoch in range(start_epoch+1, args.epochs+1):
            train_loss, train_accuracy, train_auc = train_akt(
            model, optimizer, train_loader,  label='Train')
            scheduler.step()
            valid_loss, valid_accuracy, valid_auc = test_akt(
            model, optimizer, valid_loader, label='Valid')

            print(' Epoch: {:04d}'.format(epoch),
                    ' loss_train: {:.5f}'.format(train_loss),
                    ' auc_train: {:.5f}'.format(train_auc),
                    ' acc_train: {:.5f}'.format(train_accuracy),
                    ' loss_val: {:.5f}'.format(valid_loss),
                    ' auc_val: {:.5f}'.format(valid_auc),
                    ' acc_val: {:.5f}'.format(valid_accuracy),
                    ' lr: {:.5f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

            if valid_auc > best_valid_auc:
                best_valid_auc = valid_auc
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict':scheduler.state_dict(),
                            'loss': train_loss,
                            'auc': valid_auc,
                            }, model_file)

                print(' Epoch: {:04d}'.format(epoch),
                    ' loss_train: {:.5f}'.format(train_loss),
                    ' auc_train: {:.5f}'.format(train_auc),
                    ' acc_train: {:.5f}'.format(train_accuracy),
                    ' loss_val: {:.5f}'.format(valid_loss),
                    ' auc_val: {:.5f}'.format(valid_auc),
                    ' acc_val: {:.5f}'.format(valid_accuracy), file = log)
                log.flush()

            if epoch-best_epoch > 40:
                print('best epoch is ', best_epoch)
                break
    
    elif args.mode == 1:
        log, save_dir = save_test(args.save_dir)

        id = kt_dataset.get_id()
        st_len = kt_dataset.get_stu_len()
        point = [args.start_point, kt_dataset.get_end_point()]

        test_akt_one_student(model, test_loader, save_dir, id, st_len)
    elif args.mode == 2:
        log, save_dir = save_test(args.save_dir)

        id = kt_dataset.get_id()
        st_len = kt_dataset.get_stu_len()
        point = [args.start_point, kt_dataset.get_end_point()]

        test_akt_one_student_knowledge_state(model, test_loader, save_dir, id, st_len)

    if log is not None:
        print(save_dir)
        log.close()
