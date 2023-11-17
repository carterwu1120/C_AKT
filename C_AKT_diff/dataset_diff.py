import os
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KTDataset(Dataset):
    def __init__(self, data_dir, data_file, data_pc_file, response_count_file, student_i, mode):
        super(KTDataset, self).__init__()
        self.features = None
        self.questions = None
        self.diffculty = None
        self.answers = None

        self.data_dir = data_dir
        self.data_file = data_file
        self.data_pc_file = data_pc_file
        self.response_count_file = response_count_file
        self.end_point = None
        self.student_i = student_i
        self.id = None
        self.student_i_len = None

        self.input_dim = None
        self.hidden_dim = None
        self.output_dim = None

        self.student_num = None

        self.mode = mode

        self.load_dataset()

    def get_id(self):
        return self.id

    def get_end_point(self):
        return self.end_point
    
    def get_dim(self):
        return [self.input_dim, self.hidden_dim, self.output_dim]
    
    def get_stu_num(self):
        return self.student_num
    
    def get_stu_len(self):
        return self.student_i_len

    def load_dataset(self):
        dataset_path = os.path.join(self.data_dir, self.data_file)
        df = pd.read_csv(dataset_path) # include user_id, problem_id, problem_number, time_done, time_taken, correct, count_attempts, hint_used, concept_id

        if self.mode == 1: # test mode
            dataset_pc_path = os.path.join(self.data_dir, self.data_pc_file)
            response_count_path = os.path.join(self.data_dir, self.response_count_file)
            df_pc = pd.read_csv(dataset_pc_path) # user_id = 8888888, used to add the input dim, include user_id, problem_id, correct, concept_id
            df_count = pd.read_csv(response_count_path) # include user_id, count

            self.id = df_count['user_id'][self.student_i]
            self.end_point = int(df_count['count'][self.student_i]) - 1

            df_test = df[df['user_id'] == self.id] # only include the chosen user_id
            df = df_test

            self.student_i_len = len(df_test)
            
            # df = df.append(df_pc, ignore_index = True) # add dim
            df = df.merge(df_pc, on = ['user_id', 'problem_id', 'concept_id', 'correct', 'count_attempts'], how = 'outer') # combine with pc data
            df_test['problem'], _ = pd.factorize(df_test['problem_id'], sort = True) # transform str to int
            df_test['concept'], _ = pd.factorize(df_test['concept_id'], sort = True)

            test_correct = df_test['correct'].value_counts()

            
            print('\n')
            print('Student ID: {}'.format(self.id))
            print('Response number: {}'.format(len(df_test)))
            print('Responsed correctly : {}'.format(test_correct[1]))
            print('Responsed incorrectly: {}\n'.format(test_correct[0]))


        if 'user_id' not in df.columns:
            raise KeyError(f'The column "user_id" was not found on {dataset_path}')

        if 'problem_id' not in df.columns:
            raise KeyError(f'The column "problem_id" was not found on {dataset_path}')

        if 'concept_id' not in df.columns:
            raise KeyError(f'The column "concept_id" was not found on {dataset_path}')

        if 'correct' not in df.columns:
            raise KeyError(f'The column "correct" was not found on {dataset_path}')

        if not (df['correct'].isin([0, 1])).all():
            raise KeyError(f'The values of the column "correct" must be 0 or 1.')
        

        df.dropna(subset = ['problem_id'], inplace = True)
        df.dropna(subset = ['concept_id'], inplace = True)
        df.dropna(subset = ['count_attempts'], inplace = True)
        df = df.groupby('user_id').filter(lambda q: len(q) > 1).copy()

        df['problem'], _ = pd.factorize(df['problem_id'], sort = True)
        df['feature_with_problem'] = df['problem'] * 2 + df['correct']
        
        df['concept'], _ = pd.factorize(df['concept_id'], sort = True)
        df['feature_with_concept'] = df['concept'] * 2 + df['correct']
        

        c = 2
        df['count_attempts_concept'] = df.groupby(['user_id', 'concept_id'])['count_attempts'].cumsum()
        df['concept_cognitive_diffculty'] = 0
        df['problem_cognitive_diffculty'] = 0
        for index, row in df.iterrows():
            if row['count_attempts'] < 5:

                row['problem_cognitive_diffculty'] = c
            else:
                if row['correct'] == 0:
                    row['problem_cognitive_diffculty'] = (c-1)/row['count_attempts']
                else:
                    row['problem_cognitive_diffculty'] = 0
            if row['count_attempts_concept'] < 5:
                row['concept_cognitive_diffculty'] = c
            else:
                if row['correct'] == 0:
                    row['concept_cognitive_diffculty'] = (c-1)/row['count_attempts_concept']
                else:
                    row['concept_cognitive_diffculty'] = 0

 
        feature_list = []
        question_list = []
        diffculty_list = []
        answer_list = []
        response_num_list = []
        def get_data(series):
            feature_list.append([series['feature_with_problem'].tolist(), series['feature_with_concept'].tolist()])
            question_list.append([series['problem'].tolist(), series['concept'].tolist()])
            diffculty_list.append([series['problem_cognitive_diffculty'].tolist(), series['concept_cognitive_diffculty'].tolist()])
            answer_list.append(series['correct'].eq(1).astype('int').tolist())
            response_num_list.append(series['correct'].shape[0])

        df.groupby('user_id').apply(get_data)
        self.student_num = len(response_num_list)
        max_response = np.max(response_num_list)
        question_feature_num = int(df['feature_with_problem'].max() + 1) # input dim
        question_num = int(df['problem'].max() + 1) # output dim
        print('Number of problems: ', question_num)
        self.hidden_dim = 128
        if self.mode == 0:
            print(' Number of students: ', self.student_num)
            print(' Maximum number of responses: ', max_response)
        
        self.features = feature_list
        self.questions = question_list
        self.diffculty = diffculty_list
        self.answers = answer_list
        self.input_dim = question_feature_num
        self.output_dim = question_num

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.diffculty[index], self.answers[index]

    def __len__(self):
        return len(self.features)

def pad_collate_akt(batch):
    (features, questions, diffculty, answers) = zip(*batch)
    # print(features[0][0])
    features_problem = []
    features_concept = []
    for fea in features:
        features_problem.append(fea[0])
        features_concept.append(fea[1])
    questions_problem = []
    questions_concept = []
    for ques in questions:
        questions_problem.append(ques[0])
        questions_concept.append(ques[1])
    diffculty_problem = []
    diffculty_concept = []
    for diff in diffculty:
        diffculty_problem.append(diff[0])
        diffculty_concept.append(diff[1])
    if len(features_problem[0]) < 495:
        features_problem[0].extend([0] * (495-len(features_problem[0])))
        features_concept[0].extend([0] * (495-len(features_concept[0])))
        questions_problem[0].extend([0] * (495-len(questions_problem[0])))
        questions_concept[0].extend([0] * (495-len(questions_concept[0])))
        diffculty_problem[0].extend([0] * (495-len(diffculty_problem[0])))
        diffculty_concept[0].extend([0] * (495-len(diffculty_concept[0])))
        answers[0].extend([-1] * (495-len(answers[0])))
    
        
    features_problem = [torch.LongTensor(feat) for feat in features_problem]
    features_concept = [torch.LongTensor(feat) for feat in features_concept]
    questions_problem = [torch.LongTensor(qt) for qt in questions_problem]
    questions_concept = [torch.LongTensor(qt) for qt in questions_concept]
    diffculty_problem = [torch.DoubleTensor(diff) for diff in diffculty_problem]
    diffculty_concept = [torch.DoubleTensor(diff) for diff in diffculty_concept]

    answers = [torch.LongTensor(ans) for ans in answers]
    
    feature_problem_pad = pad_sequence(features_problem, batch_first = True, padding_value = 0)
    feature_concept_pad = pad_sequence(features_concept, batch_first = True, padding_value = 0)
    question_problem_pad = pad_sequence(questions_problem, batch_first = True, padding_value = 0)
    question_concept_pad = pad_sequence(questions_concept, batch_first = True, padding_value = 0)
    diffculty_problem_pad = pad_sequence(diffculty_problem, batch_first = True, padding_value = 0)
    diffculty_concept_pad = pad_sequence(diffculty_concept, batch_first = True, padding_value = 0)
    
    answer_pad = pad_sequence(answers, batch_first = True, padding_value = -1)
    return feature_problem_pad.to(device), feature_concept_pad.to(device), question_problem_pad.to(device), question_concept_pad.to(device), diffculty_problem_pad.to(device), diffculty_concept_pad.to(device), answer_pad.to(device)
