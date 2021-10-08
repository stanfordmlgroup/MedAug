import pandas as pd
import random

from PIL import Image
from torch.utils.data import Dataset
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import glob


class PatientPositivePairDataset(Dataset):
    def __init__(self, csv_path, augmentations, same_patient=True, same_study=False,
                 diff_study=False, same_laterality=False, diff_laterality=False,
                 same_disease=False):
        """
        Args:
            csv_path (string): path to csv file, with columns:
                file_path: path to image file
                patient: patient id
                study: study number
                laterality: frontal/lateral
                disease: downstream disease label

            augmentations: composed torchvision transformations
            same_patient (bool): if True, positive pairs come from same patient
            same_study (bool): if True, positive pairs come from same study
            diff_study (bool): if True, positive pairs come from disticnt study
            if same_study and diff_study both False: positive pairs come from same patient regardless of study number

            same_laterality (bool): if True, positive pairs come from same laterality
            diff_laterality (bool): if True, positive pairs come from distinct laterality
            if same_laterality and diff_laterality both False: positive pairs come from same patient regardless of laterality

            same_disease (bool): (cheating using underlying labels): if True, images in positive pair must have the same downstream label
        """

        self.augmentations = augmentations
        self.same_patient = same_patient
        self.same_study = same_study
        self.diff_study = diff_study
        self.same_laterality = same_laterality
        self.diff_laterality = diff_laterality
        self.same_disease = same_disease

        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        query_file_path = self.df.at[idx, 'file_path']
        patient = self.df.at[idx, 'patient']
        curr_study = self.df.at[idx, 'study']
        curr_laterality = self.df.at[idx, 'laterality']
        curr_disease = self.df.at[idx, 'disease']
        df_cpy = self.df.copy()

        if self.same_patient:
            poss_key_paths = df_cpy.loc[df_cpy['patient'] == patient]

            # meta info
            query_id = int(patient[-5:])
            query_study = int(curr_study[-1])
            query_lat = 0 if curr_laterality[0] == 'f' else 1

            if self.same_study:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['study'] == curr_study]
            if self.diff_study:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['study'] != curr_study]
            if self.same_laterality:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['laterality']
                                                    == curr_laterality]
            if self.diff_laterality:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['laterality']
                                                    != curr_laterality]
            if self.same_disease:
                poss_key_paths = poss_key_paths.loc[poss_key_paths['disease']
                                                    == curr_disease]
            poss_key_paths = poss_key_paths.reset_index(drop=True)
        else:
            poss_key_paths = df_cpy.loc[df_cpy['file_path'] == query_file_path]

        query_image = Image.open(query_file_path).convert('RGB')

        try:
            sel = random.randint(0, len(poss_key_paths) - 1)
            key_id = int(poss_key_paths.at[sel, 'patient'][-5:])
            key_study = int(poss_key_paths.at[sel, 'study'][-1])
            key_lat = 0 if poss_key_paths.at[sel,
                                             'laterality'][0] == 'f' else 1
            key_image = Image.open(
                poss_key_paths.at[sel, 'file_path']).convert('RGB')
        except:
            key_id = query_id
            key_study = query_study
            key_lat = query_lat
            key_image = query_image

        meta_info = {"id": [key_id, query_id],
                     "study": [key_study, query_study],
                     "lat": [key_lat, query_lat],
                     }

        if self.augmentations is not None:
            pos_pair = [self.augmentations(
                key_image), self.augmentations(query_image)]
        else:
            pos_pair = [key_image, query_image]

        return pos_pair, meta_info, idx
