import os
import pickle
import torch
from tqdm import tqdm  # 进度条支持
from smplx import SMPL

def process_smpl_folder(root_dir):
    # 初始化SMPL模型[7](@ref)
    smpl_model = SMPL('./Pretrained/SMPL/SMPL_FEMALE.pkl', batch_size=1).cuda()
    
    # 遍历所有smpl文件夹[5](@ref)
    for root, dirs, files in os.walk(root_dir):
        if 'smpl' in dirs:
            src_dir = os.path.join(root, 'smpl')
            dst_dir = os.path.join(root, 'smpl_fg')
            
            # 创建目标目录[5](@ref)
            os.makedirs(dst_dir, exist_ok=True)
            
            # 处理所有pkl文件
            process_pkl_files(smpl_model, src_dir, dst_dir)


def process_pkl_files(model, src_dir, dst_dir):
    pkl_files = [f for f in os.listdir(src_dir) if f.endswith('.pkl')]
    
    for filename in tqdm(pkl_files, desc=f'Processing {os.path.basename(src_dir)}'):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        
        # 加载运动数据[7](@ref)
        with open(src_path, 'rb') as f:
            motion_data = pickle.load(f)
        
        # 执行原处理逻辑
        processed_data = process_single_file(model, motion_data)
        
        # 保存处理结果[7](@ref)
        with open(dst_path, 'wb') as f:
            pickle.dump(processed_data, f)


def process_single_file(model, motion_data):
    # 原处理逻辑封装（同用户提供的代码）
    t, _ = motion_data['smpl_trans'].shape
    global_orient = torch.from_numpy(motion_data['smpl_poses'][:, :3]).reshape(-1, 3).cuda()
    body_pose = torch.from_numpy(motion_data['smpl_poses'][:, 3:]).reshape(-1, 69).cuda()
    smpl_trans = torch.from_numpy(motion_data['smpl_trans'][:, :3]).reshape(-1, 3).cuda()
    
    keypoints = model.forward(
        global_orient=global_orient.float(),
        body_pose=body_pose.float(),
        transl=smpl_trans.float(),
    ).joints.detach().reshape(t, -1, 3)[:, :24, :]

    # Y轴调整计算
    keypoints_low_point, _ = torch.min(keypoints[:, :, 1], dim=1)
    keypoints_diff = keypoints_low_point - keypoints_low_point[0:1]
    keypoints_supp1 = torch.where(keypoints_diff < 0, keypoints_diff, torch.tensor(0.0))
    keypoints_supp2 = torch.where(keypoints_diff > 0, keypoints_diff, torch.tensor(0.0))
    keypoints_supp3 = (keypoints_low_point[0:1] - 0).repeat(1, t)
    
    # 更新smpl_trans
    smpl_trans[:, 1] = smpl_trans[:, 1] + (-keypoints_supp1) - keypoints_supp2 + (-keypoints_supp3)
    
    return {
        'smpl_trans': smpl_trans.cpu().numpy(),
        'smpl_poses': motion_data['smpl_poses']
    }

if __name__ == "__main__":
    # 设置根目录
    root_folder = "/data3/yangkaixing/CustomDance/GPT/code/demo/60"  # 替换为实际路径
    process_smpl_folder(root_folder)
    print("All files processed successfully!")