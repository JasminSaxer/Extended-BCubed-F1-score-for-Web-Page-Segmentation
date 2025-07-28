import json
import pprint

from src.pixel_clustering import pixel_based_clusterings
from src.node_clustering import nodes_clustering
from src.additional_stats import get_detailed_stats_per_class, get_size_of_total_groundtruth_from_clusters
from shapely.geometry import MultiPolygon, Polygon, shape
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
    Represents a Task for processing segmentation data, performing clustering, and evaluating results.

    Attributes
    ----------
    path : str
        Directory path of the annotation JSON file.
    id : str
        Identifier for the task.
    segmentations : Segmentations
        Collection of segmentation objects.
    classification_system : str
        Name of the classification system, derived from the JSON file name.
    verbose : bool
        Verbosity flag for logging.
    path_to_mhtml : str
        Path to the associated MHTML file.
    pixel_based : bool
        Whether to use pixel-based clustering.
    node_based : bool
        Whether to use node-based clustering.
    visual_segments : dict
        Pixel-based segmentation data.
    height : int
        Height of the screenshot (if pixel-based).
    width : int
        Width of the screenshot (if pixel-based).
    hyuclusters : dict
        Node-based clustering data.
    clustering_pixel : dict
        Results of pixel-based clustering.
    clustering_nodes_all : dict
        Results of node-based clustering (all nodes).
    clustering_nodes_visible : dict
        Results of node-based clustering (visible nodes only).

    Methods
    -------
    __init__(json_file, path_to_mhtml, pixel_based=True, node_based=True, folder_to_groundtruth=None, verbose=False)
        Initializes the Task object and loads data.
    transform_to_shapely_polygons(clusters=None)
        Converts cluster data to Shapely polygons.
    get_clusters(skip_visual_nodes_only=False)
        Computes pixel-based and node-based clusterings.
    calculate_score(scoring_type, verbose=False)
        Calculates B-cubed scores and saves results.
    get_clusters_and_calculate_score_per_class(scoring_type, verbose=False, measure='F1')
        Calculates per-class scores and additional statistics.
    get_node_or_pixel_clusters()
        Returns available cluster data.
    check_data_for_prediction(clusters)
        Validates cluster data for prediction.
    check_data_for_preds_dict(clusters)
        Validates prediction data for cluster dictionaries.
    get_clusters_per_tagtype(clusters)
        Groups clusters by tag type.
    __repr__()
        Returns a formatted string representation of the Task object.
    """
    def __init__(self, json_file, path_to_mhtml, pixel_based=True, node_based=True, folder_to_additional_data=None, verbose=False):
        self.path = os.path.dirname(json_file) + '/'
                
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        self.id = data.get("id")
        self.segmentations = Segmentations()
        self.classification_system = json_file.split('_')[-1].split('.')[0]
        self.verbose = verbose
        self.path_to_mhtml = path_to_mhtml
        self.pixel_based = pixel_based
        self.node_based = node_based
        self.folder_to_additional_data = folder_to_additional_data
        
        
        # check files in additional data folder
        if self.folder_to_additional_data:
            additional_json_file, additional_html_file, additional_screenshot_file = 'none', 'none', 'none'
            folder = os.path.join(self.folder_to_additional_data, self.id)
            if os.path.exists(folder):
                additional_filenames = os.listdir(folder)
                for file in additional_filenames:
                    if file.endswith('.json') and self.classification_system in file:
                        additional_json_file = os.path.join(folder, file)
                    if file.endswith('.html'):
                        additional_html_file = os.path.join(folder, file)
                    if file.endswith('.png'):
                        additional_screenshot_file = os.path.join(folder, file)

        # get data               
        if self.pixel_based:
            self.visual_segments = data.get("segmentations", {})
            self.height = data.get("height")
            self.width = data.get("width")
            
            # get screenshot path
            self.path_to_screenshot = os.path.join(self.path, f'screenshot.png')
            if not os.path.exists(self.path_to_screenshot):
                self.path_to_screenshot = os.path.join(self.folder_to_additional_data, additional_screenshot_file)
            if not os.path.exists(self.path_to_screenshot):
                raise ValueError(f'No screenshot found for {self.id} at {self.path_to_screenshot}.')
                
        if self.node_based:
            # get html folder path
            if os.path.exists(f"{self.path}dom.html"):
                self.path_to_html = f"{self.path}dom.html"
            else:
                if self.folder_to_additional_data:
                    self.path_to_html = os.path.join(self.folder_to_additional_data, additional_html_file)
                else:
                    raise ValueError("No path to HTML file provided and no default HTML file found.")
                
            self.hyuclusters = data.get("nodes")
            
        # get ground truth from additional json file if not already in main path
        if self.folder_to_additional_data:
            gt_path = os.path.join(self.folder_to_additional_data, additional_json_file)
            if os.path.exists(gt_path):
                with open(gt_path, 'r') as file:
                    data_gt = json.load(file)

                if self.pixel_based:
                    if 'ground_truth' not in self.visual_segments:
                        # Append the new dict to the existing visual_segments dict
                        gt_segments = data_gt.get("segmentations", {})
                        if hasattr(self, "visual_segments") and isinstance(self.visual_segments, dict):
                            self.visual_segments.update(gt_segments)
                if self.node_based:
                    if 'ground_truth' not in self.hyuclusters:
                        # Append the new dict to the existing hyuclusters dict
                        gt_nodes = data_gt.get("nodes", {})
                        if hasattr(self, "hyuclusters") and isinstance(self.hyuclusters, dict):
                            self.hyuclusters.update(gt_nodes)        

    def transform_to_shapely_polygons(self, clusters=None):
        self.segmentations = Segmentations()
        data_to_transform = clusters if clusters else self.visual_segments
        # make segmentations
        for key, polygons in data_to_transform.items():
            segmentation = Segmentation()
            # if data input are shapely polygons dicts
            if len(polygons) > 0:
                if isinstance(polygons[0]['polygon'], dict):
                    for polygon_item in polygons:
                        polygon = shape(polygon_item['polygon'])
                        polygon = polygon.buffer(0)
                        if not isinstance(polygon, Polygon):
                            raise ValueError(f'{polygon.geom_type} are not supported. Please convert to Polygon.')
                        segmentation.add_polygon(polygon)
                    
                # if data input are mulitpolygon lists        
                else:
                    for multipolygon_list in polygons:
                        for polygon_item in multipolygon_list['polygon']:
                            polygon_shapely = shape({'type': 'Polygon', 'coordinates': polygon_item})
                            polygon = shape(polygon_shapely)
                            polygon = polygon.buffer(0)
                            
                            # split multipolygon into single polygons
                            if isinstance(polygon, MultiPolygon):
                                for poly in polygon.geoms:
                                    segmentation.add_polygon(poly)
                            else:
                                segmentation.add_polygon(polygon)
                   
                self.segmentations.add_segmentation(key, segmentation)               
            
    def get_clusters(self,  skip_visual_nodes_only=False):
        # make clusterings
        # print('Getting pixel based Clustering...')
        
        if self.pixel_based:
            if self.verbose:
                print('getting nodes pixel clustering...')
            
            self.transform_to_shapely_polygons()
                
            self.clustering_pixel = pixel_based_clusterings(self.segmentations, self.path_to_screenshot)


        # print('Getting node based Clustering...')
        # all nodes 
        if self.node_based:
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
            # check data for prediction
            if hasattr(self, 'visual_segments'):
                self.check_data_for_prediction(self.visual_segments)
            if hasattr(self, 'hyuclusters'):
                self.check_data_for_prediction(self.hyuclusters)
                
            evaluate_function = evaluate_prediction
            
        elif scoring_type == 'pairwise_agreement':
            evaluate_function = evaluate_pairwise
        else:
            raise ValueError("Invalid scoring_type. Expected 'prediction' or 'pairwise_agreement'.")
        
 
        # calculate bcubed
        bcubed_result = {}
        
        size_names = []
        # check if have pixel-based clustering 
        if hasattr(self, 'clustering_pixel'):
            cluster_keys = self.clustering_pixel['clusters'][0].keys()
            sizes = [size for size in cluster_keys if size.startswith('size')]
            size_names.extend(sizes)
        
        # check if have node-based clustering
        if hasattr(self, 'clustering_nodes_all'):
            cluster_keys = self.clustering_nodes_all['clusters'][0].keys()
            sizes = [size for size in cluster_keys if size.startswith('size')]
            size_names.extend(sizes)
        
        for size in size_names:
            
            # pixel_based
            if 'edges' in size or 'pixel' in size:
                result_name = size[5:]
                clusters_data = self.clustering_pixel
            # node_based
            else:
                result_name = size[5:] + '_all'
                clusters_data = self.clustering_nodes_all
                
                if self.clustering_nodes_visible:
                    result_name = size[5:] + '_visible_only'
                    clusters_data = self.clustering_nodes_visible
                    
            bcubed_result[result_name] = evaluate_function(clusters_data['clusters'], clusters_data['membership'], size, self.path, verbose = verbose)
           
                
            
        df_bcube = pd.DataFrame(bcubed_result)
        df_bcube.to_csv(self.path + f'{self.classification_system}_{scoring_type}_results.csv')
        
        return df_bcube
    

    def get_clusters_and_calculate_score_per_class(self, scoring_type, verbose = False, measure = 'F1'):
        # Check data for correct form
        # define based names
        node_based_name = 'node_based'
        pixel_based_name = 'pixel_based'
        
        # get clusters pixel and node based.
        clusters = self.get_node_or_pixel_clusters() # clusters has keys: 'pixel_based' and/or 'node_based' 
        
        # segmentations
        if scoring_type == 'prediction':
            self.check_data_for_preds_dict(clusters)
            evaluate_function = evaluate_prediction
            
        elif scoring_type == 'pairwise_agreement':
            evaluate_function = evaluate_pairwise
        else:
            raise ValueError("Invalid scoring_type. Expected 'prediction' or 'pairwise_agreement'.")
        
        # splitting the clusters by tagtype
        clusters_pertagtype = {}
        if self.node_based:
            clusters_pertagtype[node_based_name], node_tagtypes, node_annotaters = self.get_clusters_per_tagtype(clusters[node_based_name])
        if self.pixel_based:
            clusters_pertagtype[pixel_based_name], pixel_tagtypes, pixel_annotaters = self.get_clusters_per_tagtype(clusters[pixel_based_name])

    
        # check if both tagtypes and annotaters are the same and assign one to tagtypes and annotaters
        if self.pixel_based and self.node_based:
            if node_tagtypes != pixel_tagtypes or node_annotaters != pixel_annotaters:
                raise ValueError("Inconsistent tagtypes or annotaters between node-based and pixel-based clusters.: {}".format((node_tagtypes, pixel_tagtypes, node_annotaters, pixel_annotaters)))
            else:
                tagtypes = node_tagtypes
                annotaters = node_annotaters
        elif self.pixel_based:
            tagtypes = pixel_tagtypes
            annotaters = pixel_annotaters
        else:
            tagtypes = node_tagtypes
            annotaters = node_annotaters


        # define atomic elements
        node_based_elements = ['nodes', 'chars']
        pixel_based_elements = ['pixel', 'edges_fine', 'edges_coarse']
        # get size of the ground truth 
        atomic_elements = []
        if self.node_based:
            atomic_elements.extend(node_based_elements)
        if self.pixel_based:
            atomic_elements.extend(pixel_based_elements)

        if self.verbose:
            print(f'Calculating {scoring_type} for {atomic_elements}...')
            
                
        # get total size overall (ignoring tagtypes) for FN and FP size relative calculation 
        if scoring_type == 'prediction':
            total_size_of_ground_truth = get_size_of_total_groundtruth_from_clusters(clusters, atomic_elements, self)  
            if verbose:
                print(f'Total size of ground truth for {atomic_elements}: {total_size_of_ground_truth}')
        
        # compare the differerent classes with each other 
        annotator_pairs = list(combinations(annotaters, 2))
        
        # create results dict
        results = {ae: {f'{annotx}_{annoty}':{t:{} for t in tagtypes} for annotx, annoty in annotator_pairs} for ae in atomic_elements}
        
        additional_stats = {'FP_size_rel':{ae: {} for ae in atomic_elements}, 'FN_size_rel':{ae: {} for ae in atomic_elements}}
 
        eval_res, clustering_res = {}, {}
        for annotx, annoty in annotator_pairs:
            if verbose:
                print(f'Calculating {scoring_type} for {annotx} and {annoty}...')
                
            for t in results[atomic_elements[0]][f'{annotx}_{annoty}'].keys():
                for t_tocompare in tagtypes:
                    # get subsets of the clusters
                    if self.node_based:
                        clusters_subset = {annotx: clusters_pertagtype[node_based_name][t][annotx],
                                        annoty: clusters_pertagtype[node_based_name][t_tocompare][annoty]}
                        self.hyuclusters = clusters_subset
                    
                    if self.pixel_based:
                        clusters_subset ={annotx: clusters_pertagtype[pixel_based_name][t][annotx],
                                        annoty: clusters_pertagtype[pixel_based_name][t_tocompare][annoty]}
                        
                        self.transform_to_shapely_polygons(clusters_subset)
                        
                    # check for empty clusters (both clustersubset should be the same)
                    none_predicted = len(clusters_subset[annotx]) == 0
                    none_groundtruth = len(clusters_subset[annoty]) == 0
                    
                    # if none predicted and none groundtruth continue                                          
                    if none_predicted and none_groundtruth:
                        continue
                    
                    if scoring_type == 'prediction':
                        # if groundtruth or predicted is empty set to 0
                        if none_groundtruth or none_predicted:
                            for ae in atomic_elements:
                                results[ae][f'{annotx}_{annoty}'][t][t_tocompare] = 0
                    
                    # make the clustering (bcubed based)
                    if self.node_based:
                        clustering_res[node_based_name] = nodes_clustering(self, node_visible_only=False)
                        
                    if self.pixel_based:
                        clustering_res[pixel_based_name] = pixel_based_clusterings(self.segmentations, self.path_to_screenshot)

                    # print(f'Calculating {scoring_type} for {annotx} and {annoty} with tagtype {t} and {t_tocompare}...')
                    for atomic_element in atomic_elements:
                        if atomic_element in node_based_elements:
                            based = node_based_name
                        elif atomic_element in pixel_based_elements:
                            based = pixel_based_name
                        else:
                            raise ValueError(f'Atomic element {atomic_element} not supported!')
                        
                        at_ele_clusters = clustering_res[based]['clusters']
                        at_ele_membership = clustering_res[based]['membership']
                        
                        # if either none in groundtruth or predicted skip the evaluation (it is set to 0)
                        if not none_groundtruth and not none_predicted:
                            eval_res[atomic_element] = evaluate_function(at_ele_clusters, at_ele_membership,
                            f'size_{atomic_element}', self.path, verbose=verbose)
                                     
                            # get measure
                            res_measure = eval_res[atomic_element].get(measure, None)
                            if res_measure is None:
                                raise ValueError(f'Measure {measure} not implemente for {scoring_type}! Try one of: {eval_res[atomic_element].keys()}')
                                
                            else:
                                results[atomic_element][f'{annotx}_{annoty}'][t][t_tocompare] = res_measure
                        
                        # get additional stats for prediction
                        if scoring_type == 'prediction':
                            # only for comparing the same class
                            if t == t_tocompare:
                                if verbose:
                                    print(f'Calculating with tagtype {t_tocompare}...')
                                add_stats = get_detailed_stats_per_class(at_ele_clusters, at_ele_membership,
                                            atomic_element, total_size_of_ground_truth, none_predicted, none_groundtruth, verbose=verbose)
                                additional_stats['FP_size_rel'][atomic_element][t_tocompare] = add_stats['FP_size_rel']
                                additional_stats['FN_size_rel'][atomic_element][t_tocompare]= add_stats['FN_size_rel']
                        
                        
        res_per_class = {}
        path_confmatrix = self.path + f'{self.classification_system}_{measure}_{scoring_type}_confmatrix/'
        if not os.path.exists(path_confmatrix):
            os.makedirs(path_confmatrix)
            
        for atomic_element in atomic_elements:
            # Calculate the mean of the matrices in results
            mean = get_mean(results[atomic_element], tagtypes)                    
            mean.to_csv(path_confmatrix + f'{atomic_element}.csv')
            # Extract diagonal values (i.e., where predicted tagtype == ground truth tagtype)
            diagonal = mean.values.diagonal()
            res_per_class[atomic_element] = pd.Series(diagonal, index=tagtypes, name=f'{atomic_element}')
        
        df_res_per_class = pd.DataFrame(res_per_class)
        df_res_per_class.to_csv(self.path + f'{self.classification_system}_{measure}_{scoring_type}_results_per_classes.csv')
            
        if scoring_type == 'prediction':
            # get mean of additional stats
            mean_fp_fn = pd.DataFrame(
                {
                    f'FP_{ae}_relativ': additional_stats['FP_size_rel'].get(ae, {}) for ae in atomic_elements
                } | {
                    f'FN_{ae}_relativ': additional_stats['FN_size_rel'].get(ae, {}) for ae in atomic_elements
                }
            ).T
            mean_fp_fn.index.name = 'statistic'
            
            path_meanfpfn = self.path + f'{self.classification_system}_FP_FN_rel_{scoring_type}_results_per_classes.csv'
            
            if os.path.exists(path_meanfpfn):
                existing_df = pd.read_csv(path_meanfpfn, index_col=0)
                # Overwrite rows if they exist, else append new
                for idx, row in mean_fp_fn.iterrows():
                    if idx in existing_df.index:
                        if verbose:
                            print(f'Overwriting existing row for {idx} in path:\n  {path_meanfpfn}')
                        existing_df.loc[idx] = row
                    else:
                        existing_df.loc[idx] = row
                mean_fp_fn = existing_df
            
            if verbose:
                print(mean_fp_fn)
            mean_fp_fn.to_csv(self.path + f'{self.classification_system}_FP_FN_rel_{scoring_type}_results_per_classes.csv')

    def get_node_or_pixel_clusters(self):
        clusters = {}
        if self.node_based:
            clusters['node_based'] = self.hyuclusters
        
        if self.pixel_based:
            clusters['pixel_based'] = self.visual_segments
        
        return clusters
    
    def check_data_for_prediction(self, clusters):
        
        # check clusters for correct form of keys
        keys = list(clusters.keys())
        if keys[0] != 'predicted':
            raise ValueError('First key in nodes data has to be "predicted".')
        if keys[1] != 'ground_truth':
            raise ValueError('Second key in nodes data has to be "ground_truth".')
    
    
    def check_data_for_preds_dict(self, clusters):
        """
        Checks and validates prediction data for different types of cluster dictionaries.
        This function inspects the provided `clusters` dictionary for the presence of
        'pixel_based' and 'node_based' keys. If found, it validates the corresponding
        cluster data using `check_data_for_prediction`. If neither key is present,
        it validates the entire `clusters` dictionary.
        Args:
            clusters (dict): Dictionary containing cluster data, possibly with
                'pixel_based' and/or 'node_based' keys.
        """
        
        pixel_based_clusters = clusters.get('pixel_based', False)
        node_based_clusters = clusters.get('node_based', False)
        if pixel_based_clusters:
            self.check_data_for_prediction(pixel_based_clusters)
        if node_based_clusters:
            self.check_data_for_prediction(node_based_clusters)
        if not pixel_based_clusters and not node_based_clusters:
            self.check_data_for_prediction(clusters)
        
    def get_clusters_per_tagtype(self, clusters):
            
        tagtypes = sorted(list(set([s['tagType'] for segments in clusters.values() for s in segments])))

        if len(tagtypes) == 0:
            return 
        
        clusters_bytagtype = {t:{} for t in tagtypes}
        for tagtype in tagtypes:
            clusters_bytagtype[tagtype] = {}
            for annotator in clusters:
                clusters_bytagtype[tagtype][annotator] = [segment for segment in clusters[annotator] if segment['tagType'] == tagtype]

        return clusters_bytagtype, tagtypes, clusters.keys()
        
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
    
def get_mean(results, tagtypes):
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
    return mean
                    