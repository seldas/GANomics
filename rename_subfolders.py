import os

def rename_subfolders(directory):
    for filename in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, filename)):
            parts = filename.split('_')
            if len(parts) == 3 and parts[0] == 'SampleSize':
                size = parts[1]
                run = parts[2]
                new_name = f'NB_Size_{size}_run_{run}'
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))
                print(f'Renamed {filename} to {new_name}')

if __name__ == '__main__':
    directory = 'results_old/NB_results'
    rename_subfolders(directory)