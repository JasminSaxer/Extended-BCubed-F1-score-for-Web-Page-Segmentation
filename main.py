import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool
from src.classes import Task
import pandas as pd

def process_folder(folder, file_postfix, operation, per_class, verbose, path_to_mhtml='', measure='F1', node_based=True, pixel_based=True, add_visible_nodes=False, folder_to_additional_data=None):
    path_annotations = folder + f'/annotations_{file_postfix}.json'

    # default both pixel_based and node_based are True
    if path_to_mhtml == '' and node_based and add_visible_nodes:
        raise ValueError(f'Path to HTML is: {path_to_mhtml}. You have to define the path to mhtml for node based calculations with visible nodes only!')
    
    task = Task(path_annotations, pixel_based=pixel_based, node_based=node_based, path_to_mhtml=path_to_mhtml, folder_to_additional_data=folder_to_additional_data, verbose = verbose)

    if per_class:
        task.get_clusters_and_calculate_score_per_class(operation, verbose, measure=measure)
    else:
        skip_visible_nodes = not add_visible_nodes              
        task.get_clusters(skip_visual_nodes_only=skip_visible_nodes)
        task.calculate_score(operation, verbose)


def get_mean_overall(main_folder, file_postfix, operation, results_folder):
    folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    df_fb3 = {}
    df_max = {}

    
    for folder in folders:
        df = pd.read_csv(folder + f'/{file_postfix}_{operation}_results.csv', index_col=0)
        pid = folder.split('/')[-1]
        df_fb3[pid] = df.loc['F1'].to_dict()
        df_max[pid] = df.loc['MaxPR'].to_dict()

    df_fb3 = pd.DataFrame(df_fb3).T
    df_max = pd.DataFrame(df_max).T



    # Create a new DataFrame to store the averages of Fb3 and Max
    results = pd.DataFrame({
        'Fb3_mean': df_fb3.mean(),
        'Max_mean': df_max.mean()
    }).T
    
    print(results.round(3))
    path= os.path.join(results_folder, f'allfolders_{file_postfix}_{operation}.csv')
    results.to_csv(path)
    
def get_mean_over_classes(main_folder, classification, measure, scoring_type, results_folder):
    folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    res_allfolders_classes = {}
    for atomic_element in ['nodes', 'chars', 'pixel', 'edges_fine', 'edges_coarse']:
        df_sum = pd.DataFrame()
        devide = df_sum.notna().astype(int)
        df_sum = df_sum.fillna(0)

        for folder in tqdm(folders):
            confmatrix_folder = f'{classification}_{measure}_{scoring_type}_confmatrix'
            file_name = f'{atomic_element}.csv'
            path = os.path.join(folder,confmatrix_folder, file_name)
            
            if os.path.exists(path):
                df = pd.read_csv(path, index_col=0)
                devide = devide.add(df.notna().astype(int), fill_value=0)
                df_sum  = df_sum.add(df.fillna(0), fill_value=0)
            else:
                print(f'{path} does not exist.')
            
        mean = df_sum / devide
        res_path_conf_matrix = os.path.join(results_folder, f'allfolders_{classification}_{measure}_{scoring_type}_confmatrix')
        os.makedirs(res_path_conf_matrix, exist_ok=True)
        mean_path = os.path.join(res_path_conf_matrix, f'{atomic_element}.csv')
        mean.to_csv(mean_path)
        
        # get only diagonal values
        diagonal = mean.values.diagonal()
        res_allfolders_classes[atomic_element] = pd.Series(diagonal, index = mean.index)

    df_res_per_class = pd.DataFrame(res_allfolders_classes)
    df_res_per_class.loc['Mean'] = df_res_per_class.mean()
    print(df_res_per_class.round(3))
    res_path = os.path.join(results_folder, f'allfolders_{classification}_{measure}_{scoring_type}_results_per_classes.csv')
    df_res_per_class.to_csv(res_path)
        
        
def get_mean_FN_FP(main_folder, file_postfix, operation, results_folder):
    folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]
    df_all = pd.DataFrame()
    
    for folder in folders:
        df = pd.read_csv(folder + f'/{file_postfix}_FP_FN_rel_{operation}_results_per_classes.csv', index_col=0)
        df_all = pd.concat([df_all, df])
    
    df_all = df_all.groupby(df_all.index).mean()
    
    print(df_all.round(3))    
    # Save the results to a CSV file
    path = os.path.join(results_folder, f'allfolders_{file_postfix}_FP_FN_rel_{operation}_results_per_classes.csv')
    df_all.to_csv(path)        
    
def process_folder_wrapper(args):
    return process_folder(*args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
        Calculates the extended B-Cubed score for Web Page Segmentation in Python, adapted from the R code available at:
        https://github.com/webis-de/cikm20-web-page-segmentation-revisited-evaluation-framework-and-dataset
        
        Example usage for pairwise agreement:
        python main.py --folder_path test_data/test_data_pairwise --file_postfix 'FC' --operation 'pairwise_agreement' --pixel_based --node_based
    
        Example usage for prediction:
        python main.py --folder_path test_data/test_data_prediction --file_postfix 'FC' --operation 'prediction'        
    """, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('--folder_path', type=str, required=True, help='The path to the folder containing the data')
    parser.add_argument('--file_postfix', type=str, required=False, default='', help='The name where files are stored : annotations_{file_postfix}.json (default: '')')
    parser.add_argument('--operation', type=str, required=True, choices=['prediction', 'pairwise_agreement'], help='The operation to perform (prediction or pairwise_agreement)')
    parser.add_argument('--path_to_mhtml', type=str, default='', help='The path to the folder containing the mhtml files. Needed only for getting visual only nodes.')
    parser.add_argument('--per_class', action='store_true', help='Calculate score per class')
    parser.add_argument('--measure', type=str, default='F1', choices=['F1', 'MaxPR', 'Precision', 'Recall'] ,help='The measure to use for per class calculation (default: F1). Can use F1, MaxPR and Recall or Precision for predicition only.')
    parser.add_argument('--use_pool', action='store_true', help='Use multiprocessing pool to process folders')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes to use in the pool (default: 4)')
    parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    parser.add_argument('--pixel_based', action='store_true', help='Use pixel based atomic element')
    parser.add_argument('--node_based', action='store_true', help='Use node based atomic element')
    parser.add_argument('--add_visible_nodes', action='store_true', help='Add measuring only visible nodes.')
    parser.add_argument('--folder_to_additional_data', type=str, default=None, help='Path to the additional data folder for static files (dom.html, screenshot.png, ground_truth.json).')

    args = parser.parse_args()

    folders = [f.path for f in os.scandir(args.folder_path) if f.is_dir()]

    if not args.node_based and not args.pixel_based:
        args.pixel_based = True

    print(f'''
    Processing folder: {args.folder_path}
    With options: 
        File Postfix \t\t\t {args.file_postfix}
        Operation \t\t\t {args.operation}
        Calculations per Class \t\t {args.per_class}
        Scoring Measure \t\t {args.measure}
        Atomic Element node based \t {args.node_based}
        Atomic Element pixel based \t {args.pixel_based}\n''')
    
    if args.use_pool:
        pool = Pool(processes=args.processes)
        for _ in tqdm(pool.imap_unordered(process_folder_wrapper, [(folder, args.file_postfix, args.operation, args.per_class, args.verbose, args.path_to_mhtml, args.measure, args.node_based, args.pixel_based, args.add_visible_nodes) for folder in folders]), total=len(folders)):
            pass
    else:
        for folder in tqdm(folders):
            process_folder(folder, 
                           args.file_postfix, 
                           args.operation, 
                           args.per_class, 
                           args.verbose, 
                           args.path_to_mhtml, 
                           args.measure, 
                           node_based=args.node_based, pixel_based=args.pixel_based, add_visible_nodes = args.add_visible_nodes, 
                           folder_to_additional_data = args.folder_to_additional_data)

    results_folder = os.path.join(os.path.dirname(args.folder_path), f'{os.path.basename(args.folder_path)}_results')
    os.makedirs(results_folder, exist_ok=True)
    
    if not args.per_class:
        get_mean_overall(args.folder_path, args.file_postfix, args.operation, results_folder)
    else:
        get_mean_over_classes(args.folder_path, args.file_postfix, args.measure, args.operation, results_folder)
        if args.operation == 'prediction':
            get_mean_FN_FP(args.folder_path, args.file_postfix, args.operation, results_folder)