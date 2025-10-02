from torch.utils.data import Dataset
import os
import numpy as np
import torch
import mne
import sys
import open3d as o3d
import pandas as pd

class AllDataFeatureTwoEEG(Dataset):
    
    def __init__(self, data_path, sub_list, train=True, time_len=250, test_mean=True, aug_data=False, point_path='', num_classes=72):
        self.data_path = data_path
        
        # Charless: 10/2 新增 classes
        self.num_classes = num_classes

        self.sub_list = sub_list
        self.train = train
        self.time_len = time_len
        self.test_mean = test_mean
        self.aug_data = aug_data
        all_name_list = sorted(os.listdir(self.data_path + 'video_new/'))
        self.name_list = []
        if train:
            remove_name = ['08', '09']
        else:
            remove_name = ['00', '01', '02', '03', '04', '05', '06', '07']
        for name in all_name_list:
            if name[-4:] != '.mp4':
                continue
            if name[-6:-4] in remove_name:
                continue
            self.name_list.append(name[:-4])
        excel_file = self.data_path + 'color_label.xlsx'
        read_df = pd.read_excel(excel_file)
        name2color_label = {}
        for index, row in read_df.iterrows():
            name2color_label[row['name']] = int(row['label'])
        
        self.name_list = np.array(self.name_list).reshape(72, -1)

        # Charless 10/2 新增：篩選前 num_classes 個類別
        if num_classes < 72:
            print(f"[Dataset] Selecting first {num_classes} classes (indices 0-{num_classes-1})")
            self.name_list = self.name_list[:num_classes]

        self.point_data = np.zeros((self.name_list.shape[0], self.name_list.shape[1], 8192, 6))
        self.color_label = np.zeros((self.name_list.shape[0], self.name_list.shape[1]))

        if point_path == '':
            point_path = self.data_path + 'point_cloud_simple/'
            for ii in range(0, self.name_list.shape[0]):
                for jj in range(0, self.name_list.shape[1]):
                    self.point_data[ii, jj] = np.load(point_path + self.name_list[ii][jj][3:] + '.npy')
                    self.color_label[ii, jj] = name2color_label[self.name_list[ii][jj][3:]]
            for ii in range(0, self.point_data.shape[0]):
                for jj in range(0, self.point_data.shape[1]):
                    self.point_data[ii, jj] = self.pc_norm(self.point_data[ii, jj])
        else:
            # import pdb;pdb.set_trace()
            app_ss = point_path.split('-')[-1][:-1]
            name_app_dir = {}
            for name_one in os.listdir(point_path):
                if '-' not in name_one:
                    continue
                name_two = name_one.split('-')[1]
                name_app_dir[name_two[3:]] = name_two[:3]
            # 同時讀取 -0..-4 的 PLY 版本，作為測試時的 5 個 trials
            K = 5
            self.point_data_k = np.zeros((self.name_list.shape[0], self.name_list.shape[1], K, 8192, 6))
            for ii in range(0, self.name_list.shape[0]):
                for jj in range(0, self.name_list.shape[1]):
                    true_points = np.load(self.data_path + 'point_cloud_simple/' + self.name_list[ii][jj][3:] + '.npy')
                    base = f'{point_path}{app_ss}-{name_app_dir.get(self.name_list[ii][jj][3:], "")}'+f'{self.name_list[ii][jj][3:]}'
                    self.color_label[ii, jj] = name2color_label[self.name_list[ii][jj][3:]]
                    found_any = False
                    for kk in range(K):
                        cand = f'{base}-{kk}.ply'
                        if os.path.exists(cand):
                            point_cloud = o3d.io.read_point_cloud(cand)
                            pts = np.asarray(point_cloud.points)
                            if pts.shape[0] != 8192:
                                raise ValueError(f'PLY size != 8192: {cand} got {pts.shape}')
                            self.point_data_k[ii, jj, kk, :, :3] = pts
                            self.point_data_k[ii, jj, kk, :, 3:] = true_points[:, 3:]
                            found_any = True
                    if not found_any:
                        raise FileNotFoundError(f'No PLY found for {base}-[0..{K-1}].ply')
        self.eeg_data, self.eeg_data2 = self.load_eeg()

        # Charless 10/2 新增：篩選 EEG data
        if num_classes < 72:
            self.eeg_data = self.eeg_data[:, :num_classes, :, :, :, :]
            self.eeg_data2 = self.eeg_data2[:, :num_classes, :, :, :, :]
        
        self.cls_num = num_classes  # 改這行

        if not self.train:
            if self.test_mean:
                self.eeg_data = np.mean(self.eeg_data, axis=3, keepdims=True)
                self.eeg_data2 = np.mean(self.eeg_data2, axis=3, keepdims=True)
                ply_trials = self.point_data_k.shape[2] if hasattr(self, "point_data_k") else 1
                self.obj_num, self.trails_num = 2, ply_trials
            else:
                eeg_trials = self.eeg_data.shape[3]
                ply_trials = self.point_data_k.shape[2] if hasattr(self, "point_data_k") else 1
                # 若存在多個 PLY 版本，優先使用其作為 trials；否則回退到 EEG trials
                self.obj_num, self.trails_num = 2, (ply_trials if ply_trials > 1 else eeg_trials)
        else:
            self.obj_num, self.trails_num = 8, 2
        
        clip_feature_name = self.data_path + 'clip_feature.pth'
        clip_feature_gray_name = self.data_path + 'clip_feature_gray.pth'
        # import pdb;pdb.set_trace()
        self.clip_features = torch.load(clip_feature_name)
        self.clip_features_gray = torch.load(clip_feature_gray_name)
        for key in self.clip_features.keys():
            one_fea = self.clip_features[key]['point']
            # one_fea = (one_fea - one_fea.min()) / (one_fea.max()- one_fea.min()) - 0.5
            one_fea = one_fea / 8.0
            self.clip_features[key]['point'] = one_fea
        for key in self.clip_features_gray.keys():
            one_fea = self.clip_features_gray[key]['point']
            # one_fea = (one_fea - one_fea.min()) / (one_fea.max()- one_fea.min()) - 0.5
            one_fea = one_fea / 8.0
            self.clip_features_gray[key]['point'] = one_fea
        print(f'name:{self.name_list.shape}, point: {self.point_data.shape}, eegdata: {self.eeg_data.shape}, eegdata2:{self.eeg_data2.shape}')
        for ii in range(0, self.name_list.shape[0]):
            for jj in range(0, self.name_list.shape[1]):
                if self.name_list[ii, jj][3:] not in self.clip_features.keys():
                    print(self.name_list[3:])
        self.txt_features = torch.zeros((self.name_list.shape[0], self.name_list.shape[1], 1024))
        self.color_video_features = torch.zeros((self.name_list.shape[0], self.name_list.shape[1], 1024))
        self.color_point_features = torch.zeros((self.name_list.shape[0], self.name_list.shape[1], 768))
        self.gray_video_features = torch.zeros((self.name_list.shape[0], self.name_list.shape[1], 1024))
        self.gray_point_features = torch.zeros((self.name_list.shape[0], self.name_list.shape[1], 768))
        for ii in range(0, self.name_list.shape[0]):
            for jj in range(0, self.name_list.shape[1]):
                name = self.name_list[ii, jj]
                self.txt_features[ii, jj] = self.clip_features[name[3:]]['text']
                self.color_video_features[ii, jj] = self.clip_features[name[3:]]['video']
                self.color_point_features[ii, jj] = self.clip_features[name[3:]]['point']
                self.gray_video_features[ii, jj] = self.clip_features_gray[name[3:]]['video']
                self.gray_point_features[ii, jj] = self.clip_features_gray[name[3:]]['point']
        # self.pre_trail_num = self.eeg_data.shape[-1] // self.time_len
    
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        other_feature = (other_feature - 0.5) * 2
        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc
    
    def load_eeg(self, ):
        eeg_path = self.data_path + 'EEGdata/'
        eeg_data_all, eeg_data_all2 = [], []
        for sub in self.sub_list:
            if self.train:
                sub_eeg_data_name2 = f"{eeg_path}{sub}/{sub}_train_data_1s_250Hz.npy"#_1s_250Hz  _6s_100Hz
                sub_eeg_data_name = f"{eeg_path}{sub}/{sub}_train_data_6s_100Hz.npy"#_1s_250Hz  _6s_100Hz
            else:
                sub_eeg_data_name2 = f"{eeg_path}{sub}/{sub}_test_data_1s_250Hz.npy"
                sub_eeg_data_name = f"{eeg_path}{sub}/{sub}_test_data_6s_100Hz.npy"#_1s_250Hz  _6s_100Hz
            sub_eeg_data = np.load(sub_eeg_data_name)
            sub_eeg_data2 = np.load(sub_eeg_data_name2)
            eeg_data_all.append(sub_eeg_data[np.newaxis, :, :, :, :, :])
            eeg_data_all2.append(sub_eeg_data2[np.newaxis, :, :, :, :, :])
        eeg_data_all = np.concatenate(eeg_data_all, axis=0)
        eeg_data_all2 = np.concatenate(eeg_data_all2, axis=0)
        return eeg_data_all, eeg_data_all2
    
    def __len__(self, ):
        return len(self.sub_list) * self.cls_num * self.obj_num * self.trails_num
    
    def add_noise(self, eeg_data):
        # import pdb;pdb.set_trace()
        stds = eeg_data.std(dim=1, keepdim=True)
        stds[torch.isnan(stds)] = 0
        noise = torch.randn_like(eeg_data) * stds * 0.2
        return eeg_data + noise
    
    def __getitem__(self, idx):
        # import pdb;pdb.set_trace()
        sub_index, sub_other = idx // (self.cls_num * self.obj_num * self.trails_num), idx % (self.cls_num * self.obj_num * self.trails_num)
        cls_index, cls_other = sub_other // (self.obj_num * self.trails_num), sub_other % (self.obj_num * self.trails_num)
        obj_index, obj_other = cls_other // self.trails_num, cls_other % self.trails_num
        name = self.name_list[cls_index, obj_index]
        if self.aug_data and np.random.rand() > 0.75:
            eeg_data = np.mean(self.eeg_data[sub_index, cls_index, obj_index, :], axis=0)
        else:
            if self.eeg_data.shape[3] == 1:
                eeg_data = np.mean(self.eeg_data[sub_index, cls_index, obj_index, :], axis=0)
            else:
                eeg_data = self.eeg_data[sub_index, cls_index, obj_index, obj_other]
        if self.aug_data and np.random.random() > 0.4:
            eeg_data_new = self.add_noise(torch.from_numpy(eeg_data))
        else:
            eeg_data_new = torch.from_numpy(eeg_data)
        
        if self.aug_data and np.random.rand() > 0.75:
            eeg_data2 = np.mean(self.eeg_data2[sub_index, cls_index, obj_index, :], axis=0)
        else:
            if self.eeg_data2.shape[3] == 1:
                eeg_data2 = np.mean(self.eeg_data2[sub_index, cls_index, obj_index, :], axis=0)
            else:
                eeg_data2 = self.eeg_data2[sub_index, cls_index, obj_index, obj_other]
        if self.aug_data and np.random.random() > 0.4:
            eeg_data2_new = self.add_noise(torch.from_numpy(eeg_data2))
        else:
            eeg_data2_new = torch.from_numpy(eeg_data2)

        # eeg_data_new = (eeg_data_new - eeg_data_new.min()) / (eeg_data_new.max() - eeg_data_new.min()) * 2.0 - 1.0
        label = cls_index
        if hasattr(self, "point_data_k"):
            point = self.point_data_k[cls_index, obj_index, obj_other]
        else:
            point = self.point_data[cls_index, obj_index]
        one_color_label = self.color_label[cls_index, obj_index]
        txt_fea = self.clip_features[name[3:]]['text']
        color_video_fea = self.clip_features[name[3:]]['video']
        color_point_fea = self.clip_features[name[3:]]['point']
        gray_video_fea = self.clip_features_gray[name[3:]]['video']
        gray_point_fea = self.clip_features_gray[name[3:]]['point']
        
        # if self.pre_trail_num != 1:
        #     if self.train:
        #         rand_index = np.random.randint(0, self.pre_trail_num)
        #         # rand_index = 0
        #         eeg_data = eeg_data[:, rand_index * self.time_len: ((rand_index + 1) * self.time_len)]
        #     else:
        #         rand_index = 0
        #         eeg_data = eeg_data[:, rand_index * self.time_len: ((rand_index + 1) * self.time_len)]
        return {'name':name, 'eeg_data':eeg_data_new, 'eeg_data2': eeg_data2_new, 'cls_label':label, 'point_cloud':torch.from_numpy(point),
                'txt_fea':txt_fea, 'color_video_fea':color_video_fea, 'color_point_fea':color_point_fea,
                'gray_video_fea':gray_video_fea, 'gray_point_fea':gray_point_fea, 'color_label':one_color_label,
                'ply_k': obj_other if hasattr(self, "point_data_k") else 0}
