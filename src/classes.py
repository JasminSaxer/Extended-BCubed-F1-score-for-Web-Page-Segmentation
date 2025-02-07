import json
import pprint

from src.pixel_clustering import pixel_based_clusterings
from src.node_clustering import nodes_clustering
from shapely.geometry import MultiPolygon, Polygon
from src.bcubed import evaluate_pairwise, evaluate_prediction
import pandas as pd
import os
from itertools import combinations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Segmentation:
    """
    Represents a single segmentation, which contains one or more MultiPolygon geometries.
    """
    def __init__(self, polygons=None):
        # polygons should be a list of Shapely MultiPolygon objects
        self.polygons = polygons if polygons else []

    def __repr__(self):
        return f"Segmentation({self.polygons})"

    def add_polygon(self, polygon):
        self.polygons.append(polygon)

    def __len__(self):
        return len(self.polygons)
    
    def __iter__(self):
        """
        Returns an iterator over the segmentations.
        """
        return iter(self.polygons)

    
class Segmentations:
    """
    Represents a collection of segmentations, each identified by a key.
    """
    def __init__(self):
        self.segmentations = {}
        self.geometrics = None
    
    def __repr__(self):
        return f"Segmentations({self.segmentations})"

    def add_segmentation(self, key, segmentation):
        """
        Adds a segmentation to the collection.
        :param key: Identifier for the segmentation.
        :param segmentation: A Segmentation object.
        """
        if not isinstance(segmentation, Segmentation):
            raise TypeError("segmentation must be an instance of the Segmentation class")
        
        self.segmentations[key] = segmentation

    def get_segmentation(self, key):
        """
        Retrieves a segmentation by key.
        :param key: Identifier for the segmentation.
        :return: A Segmentation object or None if key does not exist.
        """
        return self.segmentations.get(key, None)

    def filter_segmentations(self, condition_function):
        """
        Filters the segmentations based on a condition function.
        :param condition_function: A function that takes a (key, segmentation) pair
                                   and returns True if it should be kept.
        :return: A new Segmentations object with filtered data.
        """
        filtered = Segmentations()
        for key, segmentation in self.segmentations.items():
            if condition_function(key, segmentation):
                filtered.add_segmentation(key, segmentation)
        return filtered
    
    def __iter__(self):
        """
        Returns an iterator over the segmentations.
        """
        return iter(self.segmentations.items())
    
    def values(self):
        """
        Returns a list of all segmentations in the collection.
        """
        return list(self.segmentations.values())
    

class Task:
    """
    A class to represent a Task that processes segmentation data and performs various clustering and evaluation operations.
    Attributes:
    -----------
    path : str
        The directory path of the JSON file containing the annotations.
    id : str
        The ID of the task.
    height : int
        The height of the screenshot.
    width : int
        The width of the screenshot.
    segmentations : Segmentations
        An instance of the Segmentations class.
    hyuclusters : dict
        A dictionary containing node data, based hyu values.
    clustering_pixel : dict
        A dictionary containing pixel-based clustering results.
    clustering_nodes_all : dict
        A dictionary containing node-based clustering results for all nodes.
    clustering_nodes_visible : dict
        A dictionary containing node-based clustering results for visible nodes only.
    hyuclusters_bytagtype : dict
        A dictionary containing clusters grouped by tag types.
        
    Methods:
    --------
    __init__(json_file):
        Initializes the Task object with data from a JSON file.
    get_clusters():
        Generates pixel-based and node-based clustering results.
    get_res_per_class(verbose=False, measure='F1'):
        Calculates pairwise results per class and saves the results to a CSV file.        
    calculate_pairwise_agreement(result_file_name='', verbose=False):
        Evaluates clustering results by pairwise agreement and saves the results to a CSV file.
    calculate_prediction_score(verbose=False):
        Evaluates prediction score results and saves the results to a CSV file.
    __repr__():
        Returns a string representation of the Task object.
    """
    
    def __init__(self, json_file, path_to_mhtml, verbose=False):
        self.path = os.path.dirname(json_file) + '/'
        
                
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        self.id = data.get("id")
        self.height = data.get("height")
        self.width = data.get("width")
        self.segmentations = Segmentations()
        self.classification_system = json_file.split('_')[-1].split('.')[0]
        self.verbose = verbose
        self.path_to_mhtml = path_to_mhtml
        
        # get data
        for key, polygons in data.get("segmentations", {}).items():
            segmentation = Segmentation()
            for polygon_list in polygons:
                for polygon_coords in polygon_list['polygon']:
                    polygon = Polygon(polygon_coords[0])
                    segmentation.add_polygon(polygon)
            self.segmentations.add_segmentation(key, segmentation)
            
        self.hyuclusters = data.get("nodes")
    
    def get_clusters(self, skip_visual_nodes_only=False):
        # make clusterings
        # print('Getting pixel based Clustering...')
        if self.verbose:
            print('getting nodes pixel clustering...')
        self.clustering_pixel = pixel_based_clusterings(self.segmentations, self.path)
        
        # print('Getting node based Clustering...')
        # all nodes 
        if self.verbose:
            print('getting nodes all clustering...')
        self.clustering_nodes_all = nodes_clustering(self, node_visible_only=False)
        # visible nodes only
        if not skip_visual_nodes_only:
            if self.verbose:
                print('getting nodes visible clustering...')
            self.clustering_nodes_visible = nodes_clustering(self, node_visible_only=True, path_to_mhtml = self.path_to_mhtml)
        else:
            self.clustering_nodes_visible = None
    
    def calculate_score(self, scoring_type, verbose = False):
        """
        Calculate the B-cubed score for the given scoring type and saves to csv.
        Parameters:
            scoring_type (str): The type of scoring to be used. Must be either 'prediction' or 'pairwise_agreement'.
            verbose (bool, optional): If True, prints detailed information during the calculation. Defaults to False.
        Raises:
            ValueError: If the first key in segmentations data is not 'predicted'.
            ValueError: If the second key in segmentations data is not 'ground_truth'.
            ValueError: If the first key in nodes data is not 'predicted'.
            ValueError: If the second key in nodes data is not 'ground_truth'.
            ValueError: If scoring_type is not 'prediction' or 'pairwise_agreement'.
        Returns:
            pd.DataFrame: A DataFrame containing the B-cubed results for different sizes and node visibility.
        """

        # Check data for correct form
        # segmentations
        if scoring_type == 'prediction':
            keys = list(self.segmentations.segmentations.keys())
            if keys[0] != 'predicted':
                raise ValueError('First key in segmentations data has to be "predicted".')
            if keys[1] != 'ground_truth':
                raise ValueError('Second key in segmentations data has to be "ground_truth".')
            
            # nodes
            keys = list(self.hyuclusters.keys())
            if keys[0] != 'predicted':
                raise ValueError('First key in nodes data has to be "predicted".')
            if keys[1] != 'ground_truth':
                raise ValueError('Second key in nodes data has to be "ground_truth".')
            
            evaluate_function = evaluate_prediction
            
        elif scoring_type == 'pairwise_agreement':
            evaluate_function = evaluate_pairwise
        
        else:
            raise ValueError("Invalid scoring_type. Expected 'prediction' or 'pairwise_agreement'.")
     
        # calculate bcubed
        bcubed_result = {}
        
        # pixel-based clustering
        size_names = ['size_edges_fine', 'size_edges_coarse', 'size_pixel']
        if verbose:
            print(f'extended Bcubed results for {self.path}:')
        for size in size_names:
            bcubed_result[size[5:]] = evaluate_function(self.clustering_pixel['clusters'], self.clustering_pixel['membership'], size, self.path, verbose = verbose)
        
        # all nodes
        size_names = ['size_nodes', 'size_chars']
        for size in size_names:
            bcubed_result[size[5:]+'_all'] = evaluate_function(self.clustering_nodes_all['clusters'], self.clustering_nodes_all['membership'], size,  self.path, verbose = verbose)
             
        
        if self.clustering_nodes_visible:
            # only visible nodes
            size_names = ['size_nodes', 'size_chars']
            for size in size_names:
                bcubed_result[size[5:]+'_visible_only'] = evaluate_function(self.clustering_nodes_visible['clusters'], self.clustering_nodes_visible['membership'], size,  self.path, verbose = verbose)
            
        df_bcube = pd.DataFrame(bcubed_result)
        
            
        df_bcube.to_csv(self.path + f'{self.classification_system}_{scoring_type}_results.csv')
        
        return df_bcube
    

    def get_clusters_and_calculate_score_per_class(self, scoring_type, verbose = False, measure = 'F1'):
        # Check data for correct form
        # segmentations
        if scoring_type == 'prediction':
            keys = list(self.segmentations.segmentations.keys())
            if keys[0] != 'predicted':
                raise ValueError('First key in segmentations data has to be "predicted".')
            if keys[1] != 'ground_truth':
                raise ValueError('Second key in segmentations data has to be "ground_truth".')
            
            # nodes
            keys = list(self.hyuclusters.keys())
            if keys[0] != 'predicted':
                raise ValueError('First key in nodes data has to be "predicted".')
            if keys[1] != 'ground_truth':
                raise ValueError('Second key in nodes data has to be "ground_truth".')
            
            evaluate_function = evaluate_prediction
            
        elif scoring_type == 'pairwise_agreement':
            evaluate_function = evaluate_pairwise
        
        else:
            raise ValueError("Invalid scoring_type. Expected 'prediction' or 'pairwise_agreement'.")
        
        tagtypes = sorted(list(set([s['tagType'] for segments in self.hyuclusters.values() for s in segments])))
        if len(tagtypes) == 0:
            return 
        
        self.hyuclusters_bytagtype = {t:{} for t in tagtypes}
        for tagtype in tagtypes:
            self.hyuclusters_bytagtype[tagtype] = {}
            for annotator in self.hyuclusters:
                self.hyuclusters_bytagtype[tagtype][annotator] = [segment for segment in self.hyuclusters[annotator] if segment['tagType'] == tagtype]

        # compare the differerent classes with each other 
        annotator_pairs = list(combinations(self.hyuclusters.keys(), 2))
        
        results = {f'{annotx}_{annoty}':{} for annotx, annoty in annotator_pairs}
        
        for annotx, annoty in annotator_pairs:
            
            results[f'{annotx}_{annoty}'] = {t:{} for t in tagtypes}
            
            for t in results[f'{annotx}_{annoty}'].keys():
                for t_tocompare in tagtypes:
                    self.hyuclusters = {annotx: self.hyuclusters_bytagtype[t][annotx],
                                        annoty: self.hyuclusters_bytagtype[t_tocompare][annoty]}

                    if len(self.hyuclusters[annotx]) == 0 and len(self.hyuclusters[annoty]) == 0:
                        continue
                    nodes_all_clusters = nodes_clustering(self, node_visible_only=False)
                    
                    eval_res = evaluate_function(nodes_all_clusters['clusters'], nodes_all_clusters['membership'],
                            'size_nodes', self.path, verbose=verbose)
                    
                    # get measure
                    res_measure = eval_res.get(measure)
                    if res_measure is None:
                        print(f'Measure {measure} not implemente for {scoring_type}! Try one of: {eval_res.keys()}')
                        
                    else:
                        results[f'{annotx}_{annoty}'][t][t_tocompare] = res_measure
                   

        # Calculate the mean of the matrices in results
        dfs = []
        for annot_pair in results:
            df_pairwise_fb3 = pd.DataFrame(results[annot_pair])
            
            df_pairwise_fb3 = df_pairwise_fb3.sort_index(axis=0).sort_index(axis=1)
            dfs.append(df_pairwise_fb3)

        sum = pd.DataFrame(index=tagtypes, columns=tagtypes)
        devide = sum.notna().astype(int)
        sum = sum.fillna(0)
        
        # get all columns of all dfs
        for df in dfs:
            # Check if columns are missing
            missing_columns = list(set(tagtypes) - set(df.columns))

            # Add missing columns with None values
            for col in missing_columns:
                df[col] = None
            
            df = df[[col for col in tagtypes]]           
            
            devide += df.notna().astype(int)
            sum += df.fillna(0)
            
        mean = sum / devide
                        
        mean.to_csv(self.path + f'{self.classification_system}_{measure}_{scoring_type}_results_per_classes_all_nodes.csv')
        
        
    def __repr__(self):
        return pprint.pformat(self.__dict__, indent=4)
    
class Cluster:
    def __init__(self, multipolygon, membership, size=1):
        """
        Initialize a Cluster object.

        :param multipolygon: A Shapely MultiPolygon object.
        :param membership: A list of boolean values representing membership.
        :param size: A numeric value representing the size of the cluster.
        """
        # Validate multipolygon
        if not isinstance(multipolygon, MultiPolygon):
            raise TypeError(f"Given geometry was not a MULTIPOLYGON, but {type(multipolygon).__name__}")
        
        # Validate membership
        if not isinstance(membership, list) or not all(isinstance(x, bool) for x in membership):
            raise TypeError(f"Given membership was not a list of booleans, but {type(membership).__name__}")
        
        # Validate size
        if not isinstance(size, (int, float)):
            raise TypeError(f"Given size was not numeric, but {type(size).__name__}")
        
        # Assign attributes
        self.multipolygon = multipolygon
        self.membership = membership
        self.size = size
    
    def __repr__(self):
        """
        String representation of the Cluster object.
        """
        return f"Cluster(multipolygon={self.multipolygon}, membership={self.membership}, size={self.size})"