import os
from tqdm import tqdm
from multiprocessing import Pool
from src.classes import Task
import pandas as pd


def process_folder(folder):
    task = Task(folder + '/annotations.json')
    task.calculate_pairwise_agreement(verbose=True)

def get_mean_overall(folder):
    
    folders = [f.path for f in os.scandir(folder) if f.is_dir()]
    df_fb3 = {}
    df_max = {}
    
    for folder in folders:
        df = pd.read_csv(folder + '/bcubed_res.csv', index_col=0)
        pid = folder.split('/')[-1]
        df_fb3[pid] = df.loc['fb3'].to_dict()
        df_max[pid] = df.loc['max'].to_dict()
    
    df_fb3 = pd.DataFrame(df_fb3).T
    df_max = pd.DataFrame(df_max).T
    
    print(df_fb3)
    print(df_max)
    
    print('Fb3:\n', df_fb3.mean().round(3))
    print('Max:\n', df_max.mean().round(3))
    
    
if __name__ == '__main__':
    
    # Pairwise agreement
    ## process all pages in the folder for pairwise agreement
    folder_path = 'test_data'
    folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    pool = Pool(processes=4)
    for _ in tqdm(pool.imap_unordered(process_folder, folders), total=len(folders)):
        pass
    
    ## get mean results over all pages
    get_mean_overall(folder_path)
    
    # Prediction score
    ## calculate prediction score for one page
    folder = 'test_data/prediction_testpage'
    task = Task(folder + '/annotations.json')
    task.calculate_prediction_score(verbose=True)