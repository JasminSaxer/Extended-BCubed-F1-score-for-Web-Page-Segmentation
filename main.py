import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from src.classes import Task
import pandas as pd

def process_folder(folder, class_system, operation, per_class, verbose, path_to_mhtml='', measure='F1'):
    path_annotations = folder + f'/annotations_{class_system}.json'
    if per_class:
        path_res = folder + f'/{class_system}_{measure}_{operation}_results_per_classes_all_nodes.csv'
        if not os.path.exists(path_res):
            task = Task(path_annotations, verbose)
            task.get_clusters_and_calculate_score_per_class(operation, verbose, measure=measure)
    else:
        if path_to_mhtml == '':
            print('You have to define the path to mhtml!')
            exit(0)
        path_res = folder + f'/{class_system}_{operation}_results.csv'
        
        if not os.path.exists(path_res):
            task = Task(path_annotations, path_to_mhtml, verbose)
            task.get_clusters(skip_visual_nodes_only=False)
            task.calculate_score(operation, verbose)


def get_mean_overall(main_folder, class_system, operation):
    folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    df_fb3 = {}
    df_max = {}

    for folder in folders:
        df = pd.read_csv(folder + f'/{class_system}_{operation}_results.csv', index_col=0)
        pid = folder.split('/')[-1]
        df_fb3[pid] = df.loc['F1'].to_dict()
        df_max[pid] = df.loc['MaxPR'].to_dict()

    df_fb3 = pd.DataFrame(df_fb3).T
    df_max = pd.DataFrame(df_max).T

    print(df_fb3)
    print(df_max)

    print('Fb3:\n', df_fb3.mean().round(3))
    print('Max:\n', df_max.mean().round(3))
    
    path_fb3 = os.path.join(os.path.dirname(main_folder), f'{class_system}_fb3_{operation}_results_allfolders.csv')
    df_fb3.to_csv(path_fb3)
    
    path_max = os.path.join(os.path.dirname(main_folder), f'{class_system}_max_{operation}_results_allfolders.csv')
    df_max.to_csv(path_max)
    
def get_mean_over_classes(main_folder, classification, measure, scoring_type):
    folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

        
    df_sum = pd.DataFrame()
    devide = df_sum.notna().astype(int)
    df_sum = df_sum.fillna(0)

    for folder in tqdm(folders):
        file_name = f'{classification}_{measure}_{scoring_type}_results_per_classes_all_nodes.csv'
        path = os.path.join(folder, file_name)
        
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            devide = devide.add(df.notna().astype(int), fill_value=0)
            df_sum  = df_sum.add(df.fillna(0), fill_value=0)
        else:
            print(f'{path} does not exist.')
        
    mean = df_sum / devide
    print(mean.round(3))   
    mean_path = os.path.join(os.path.dirname(main_folder), f'{classification}_{measure}_{scoring_type}_results_per_classes_all_nodes_allfolders.csv')
    mean.to_csv(mean_path)
    
def process_folder_wrapper(args):
    return process_folder(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process folders.')
    
    parser.add_argument('--folder_path', type=str, help='The path to the folder containing the data')
    parser.add_argument('--class_system', type=str, choices=['FC', 'MC'], help='The class system to use (FC or MC)')
    parser.add_argument('--operation', type=str, choices=['prediction', 'pairwise_agreement'], help='The operation to perform (prediction or pairwise_agreement)')
    parser.add_argument('--path_to_mhtml', type=str, default='', help='The path to the folder containing the mhtml files.')
    parser.add_argument('--per_class', action='store_true', help='Calculate score per class')
    parser.add_argument('--measure', type=str, default='F1', choices=['F1', 'MaxPR', 'Precision', 'Recall'] ,help='The measure to use for per class calculation (default: F1). Can use F1, MaxPR and Recall or Precision for predicition only.')
    parser.add_argument('--use_pool', action='store_true', help='Use multiprocessing pool to process folders')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes to use in the pool (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    
    args = parser.parse_args()

    folders = [f.path for f in os.scandir(args.folder_path) if f.is_dir()]

    if args.use_pool:
        pool = Pool(processes=args.processes)
        for _ in tqdm(pool.imap_unordered(process_folder_wrapper, [(folder, args.class_system, args.operation, args.per_class, args.verbose, args.path_to_mhtml, args.measure) for folder in folders]), total=len(folders)):
            pass
    else:
        for folder in tqdm(folders):
            process_folder(folder, args.class_system, args.operation, args.per_class, args.verbose, args.path_to_mhtml, args.measure)

    if not args.per_class:
        get_mean_overall(args.folder_path, args.class_system, args.operation)

    else:
        get_mean_over_classes(args.folder_path, args.class_system, args.measure, args.operation)