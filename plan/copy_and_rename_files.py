import os
import shutil

def copy_and_rename_files(directory):
    checkpoints_dst_root = 'dashboard/backend/results_ms/1_Training/checkpoints'
    logs_dst = 'dashboard/backend/results_ms/1_Training/logs'
    os.makedirs(logs_dst, exist_ok=True)

    for subfolder in os.listdir(directory):
        subfolder_path = os.path.join(directory, subfolder)
        if os.path.isdir(subfolder_path):
            # Create destination subfolder under checkpoints
            parts = subfolder.split('_')
            size = parts[2]
            run = parts[4]
            checkpoints_dst = os.path.join(checkpoints_dst_root, f'NB_Size_{size}_run_{run}')
            os.makedirs(checkpoints_dst, exist_ok=True)

            # Copy .pth files
            for filename in os.listdir(subfolder_path):
                if filename.startswith('latest_') and filename.endswith('.pth'):
                    src_file = os.path.join(subfolder_path, filename)
                    shutil.copy(src_file, checkpoints_dst)
                    print(f'Copied {src_file} to {checkpoints_dst}')

            # Copy and rename loss_log.txt
            src_file = os.path.join(subfolder_path, 'loss_log.txt')
            if os.path.exists(src_file):
                dst_file = os.path.join(logs_dst, f'NB_Size_{size}_run_{run}.txt')
                shutil.copy(src_file, dst_file)
                print(f'Copied and renamed {src_file} to {dst_file}')

if __name__ == '__main__':
    directory = 'results_old/NB_results'
    copy_and_rename_files(directory)