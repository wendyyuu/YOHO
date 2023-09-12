"""
Generate YOHO input for Testset.
PC*60 rotations->FCGF backbone-> FCGF Group feature for PC keypoints.
"""

import time
import numpy as np
import argparse
import open3d as o3d
import torch
from tqdm import tqdm
from utils.dataset import get_dataset_name
from utils.utils import make_non_exists_dir
from fcgf_model import load_model
from knn_search import knn_module
import MinkowskiEngine as ME
from utils.utils_o3d import make_open3d_point_cloud
from utils.utils import random_rotation_matrix
import os
from utils.misc import extract_features
#importing the module 
import logging 

class FCGFDataset():
    def __init__(self,datasets,config):
        self.points={}
        self.pointlist=[]
        self.voxel_size = config.voxel_size
        self.datasets=datasets
        self.Rgroup=np.load('./group_related/Rotation.npy')
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            for pc_id in dataset.pc_ids:
                for g_id in range(60):
                    self.pointlist.append((scene,pc_id,g_id))
                self.points[f'{scene}_{pc_id}']=self.datasets[scene].get_pc(pc_id)


    def __getitem__(self, idx):
        scene,pc_id,g_id=self.pointlist[idx]
        xyz0 = self.points[f'{scene}_{pc_id}']
        # **
        xyz0=xyz0@self.Rgroup[g_id].T
        # Voxelization
        _, sel0 = ME.utils.sparse_quantize(xyz0 / self.voxel_size, return_index=True)
        # Make point clouds using voxelized points
        pcd0 = make_open3d_point_cloud(xyz0)
        # Select features and points using the returned voxelized indices
        pcd0.points = o3d.utility.Vector3dVector(np.array(pcd0.points)[sel0])
        # Get coords
        xyz0 = np.array(pcd0.points)
        feats=np.ones((xyz0.shape[0], 1))
        coords0 = np.floor(xyz0 / self.voxel_size)
        
        return (xyz0, coords0, feats ,self.pointlist[idx])
    
    def __len__(self):
        return len(self.pointlist)




class testset_create():
    def __init__(self,config):
        self.config=config
        self.dataset_name=self.config.dataset
        self.output_dir='./data/YOHO_FCGF'
        self.origin_dir='./data/origin_data'
        self.datasets=get_dataset_name(self.dataset_name,self.origin_dir)
        self.Rgroup=np.load('./group_related/Rotation.npy')
        self.stationnums=[60,60,60,55,57,37,66,38]
        self.knn=knn_module.KNN(1)


    def collate_fn(self,list_data):
        xyz0, coords0, feats0, scenepc = list(
            zip(*list_data))
        xyz_batch0 = []
        dsxyz_batch0=[]
        batch_id = 0
        def to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            elif isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                raise ValueError(f'Can not convert to torch tensor, {x}')
        
        
        for batch_id, _ in enumerate(coords0):
            xyz_batch0.append(to_tensor(xyz0[batch_id]))
            _, inds = ME.utils.sparse_quantize(coords0[batch_id], return_index=True)
            dsxyz_batch0.append(to_tensor(xyz0[batch_id][inds]))

        coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)

        # Concatenate all lists
        xyz_batch0 = torch.cat(xyz_batch0, 0).float()
        dsxyz_batch0=torch.cat(dsxyz_batch0, 0).float()
        cuts_node=0
        cuts=[0]
        for batch_id, _ in enumerate(coords0):
            cuts_node+=coords0[batch_id].shape[0]
            cuts.append(cuts_node)

        return {
            'pcd0': xyz_batch0,
            'dspcd0':dsxyz_batch0,
            'scenepc':scenepc,
            'cuts':cuts,
            'sinput0_C': coords_batch0,
            'sinput0_F': feats_batch0.float(),
        }

    def Feature_extracting(self, data_loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(self.config.model)
        config = checkpoint['config']
        features={}
        features_gid={}
        for scene,dataset in self.datasets.items():
            if scene=='wholesetname':continue
            for pc_id in dataset.pc_ids:
                features[f'{scene}_{pc_id}']=[]
                features_gid[f'{scene}_{pc_id}']=[]

        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
            
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
         
        with torch.no_grad():
            for i, input_dict in enumerate(tqdm(data_loader)):
                starttime=time.time()
                sinput0 = ME.SparseTensor(
                        input_dict['sinput0_F'].to(device),
                        coordinates=input_dict['sinput0_C'].to(device))
                torch.cuda.synchronize()
                F0 = model(sinput0).F
            
                cuts=input_dict['cuts']
                scene_pc=input_dict['scenepc']
                for inb in range(len(scene_pc)):
                    scene,pc_id,g_id=scene_pc[inb]
                    make_non_exists_dir(f'{self.output_dir}/Testset/{self.dataset_name}/{scene}/FCGF_Input_Group_feature')
                    feature=F0[cuts[inb]:cuts[inb+1]].cpu().numpy()
                    pts=input_dict['dspcd0'][cuts[inb]:cuts[inb+1]].numpy()#*config.voxel_size
                    # **
                    Keys=self.datasets[scene].get_kps(pc_id)
                    # **
                    Keys=Keys@self.Rgroup[g_id].T
                    Keys_i=torch.from_numpy(np.transpose(Keys)[None,:,:]).cuda() #1,3,k
                    xyz_down=torch.from_numpy(np.transpose(pts)[None,:,:]).cuda() #1,3,n
                    d,nnindex=self.knn(xyz_down,Keys_i)
                    nnindex=nnindex[0,0].cpu().numpy()
                    one_R_output=feature[nnindex,:]#5000*32
                    features[f'{scene}_{pc_id}'].append(one_R_output[:,:,None])
                    features_gid[f'{scene}_{pc_id}'].append(g_id)
                    if len(features_gid[f'{scene}_{pc_id}'])==60:
                        sort_args=np.array(features_gid[f'{scene}_{pc_id}'])
                        sort_args=np.argsort(sort_args)
                        output=np.concatenate(features[f'{scene}_{pc_id}'],axis=-1)[:,:,sort_args]
                        np.save(f'{self.output_dir}/Testset/{self.dataset_name}/{scene}/FCGF_Input_Group_feature/{pc_id}.npy',output)
                        features[f'{scene}_{pc_id}']=[]
    
   
    def PC_random_rot(self,args):
        for scene, dataset in tqdm(self.datasets.items()):
            if scene in ['wholesetname']:continue
            for pc_id in tqdm(dataset.pc_ids):
                Rot_save_dir=f'{self.origin_dir}/{self.dataset_name}/'+ scene
                # add a new random_rot()
                PC = dataset.get_pc(pc_id)
                # print("PC_shape: ", np.shape(PC)) #(point_number, 3)
                R_one = random_rotation_matrix() #(3, 3)
                PC_one = PC @ R_one.T
                # print("PC_one_shape: ", np.shape(PC_one))
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(PC_one)
                # save new point cloud to {self.root}/PointCloud_rot/cloud_bin_{k}.ply
                # save the rotation matrix R_one to .npy
                # make PointCloud_rot dir
                make_non_exists_dir(f'{Rot_save_dir}/PointCloud_rot')
                # make gt2.log in PointClouds_rot dir
                # make_non_exists_dir(f'{Rot_save_dir}/PointCloud_rot/gt2.log')
                # modify
                np.save(f'{Rot_save_dir}/PointCloud_rot/Rotation_{pc_id}.npy',R_one)
                # np.save(f'{Rot_save_dir}/PointCloud_rot/Rotation_{pc_id}.npy',R_one.T)
                # np.savez(f'{Rot_save_dir}/{pc_id}_feats.ply',Rs=Random_Rs,feats=Feats)
                o3d.io.write_point_cloud(f'{Rot_save_dir}/PointCloud_rot/cloud_bin_{pc_id}.ply', pcd, write_ascii=False, compressed=False, print_progress=False)
                # print("complete rotation")
                
    def testset(self):
        print("testset func")
        for index, (name, dataset) in enumerate(self.datasets.items()):
            if name in ['wholesetname']:continue
            for pair in tqdm(dataset.pair_ids):
                pc0,pc1=pair
                Rot_save_dir=f'{self.origin_dir}/{self.dataset_name}/'+ name
                print("Rot_save_dir: ", Rot_save_dir)
                #if os.path.exists(f'{Save_list_dir}/{i*16)}.pth'):continue
                #feature readin
                R_i=np.load(f'{Rot_save_dir}/PointCloud_rot/Rotation_{pc0}.npy')
                R_j=np.load(f'{Rot_save_dir}/PointCloud_rot/Rotation_{pc1}.npy')
                T_gt = dataset.get_gt_transform(pc0,pc1) # 3 * 4
                T = np.append(T_gt[:, 3], 1)
                T_ = np.array([T])
                # print("t:", t)
                # print("t.shape:", np.shape(t))
                t = T_.reshape(-1, 1)
                # print("t:", t)
                # print("t.shape:", np.shape(t))
                # R_gt = dataset.get_gt_transform(pc0,pc1)[0:3,0:3]
                R_gt = T_gt[0:3,0:3]
                
                R = R_j @ R_gt.T @ R_i.T # from pc0 to pc1
                # print("R:", R)
                # print("R.shape:", np.shape(R))
                z = np.array([0, 0, 0])
                R_ = np.vstack([R, z])
                # print("R_:", R_)
                # print("R_.shape:", np.shape(R_))
                T_gt_new = np.hstack([R_, t])
                # print and save:
                msg = f'{pc0}  {pc1}  {self.stationnums[index - 1]}\n'
                print("Write gt2.log")
                print(name, ", ", self.stationnums[index - 1], ", ", index - 1)
                
                # msg += f'{T_gt_new}\n'
                with open(f'{Rot_save_dir}/PointCloud_rot/gt2.log','a') as f:
                    f.write(msg)
                r, c = T_gt_new.shape
                for i in range(r):
                    for j in range(c):
                        arr = T_gt_new[i, j]
                        a = f'{arr} '
                        with open(f'{Rot_save_dir}/PointCloud_rot/gt2.log','a') as f:
                            f.write(a)
                    with open(f'{Rot_save_dir}/PointCloud_rot/gt2.log','a') as f:
                        b = f'\n'
                        f.write(b)
               

    def batch_feature_extraction(self):
        dset=FCGFDataset(self.datasets,self.config)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=4, #6 is timely better(but out of memory easily)
            shuffle=False,
            num_workers=10,
            collate_fn=self.collate_fn,
            pin_memory=False,
            drop_last=False)
        self.Feature_extracting(loader)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        default='./model/Backbone/best_val_checkpoint.pth',
        type=str,
        help='path to backbone latest checkpoint (default: None)')
    parser.add_argument(
        '--voxel_size',
        default=0.025,
        type=float,
        help='voxel size to preprocess point cloud')
    parser.add_argument(
        '--dataset',
        default='demo',
        type=str,
        help='datasetname')
    args = parser.parse_args()

    testset_creater=testset_create(args)
    # print("1")
    testset_creater.PC_random_rot(args)
    # print("2")
    testset_creater.testset()
    # print("3")
    testset_creater.batch_feature_extraction()
    # print("4")
    