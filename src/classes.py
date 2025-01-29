import json
import pprint

from src.pixel_clustering import pixel_based_clusterings
from src.node_clustering import nodes_clustering
from shapely.geometry import MultiPolygon, Polygon
from src.bcubed import evaluate, evaluate_prediction
import pandas as pd


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
    def __init__(self, json_file):
        self.path = json_file[:-16]
        
                
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        self.id = data.get("id")
        self.height = data.get("height")
        self.width = data.get("width")
        self.segmentations = Segmentations()
        
        
        # get data
        for key, polygons in data.get("segmentations", {}).items():
            segmentation = Segmentation()
            for polygon_list in polygons:
                for polygon_coords in polygon_list:
                    polygon = Polygon(polygon_coords[0])
                    # buffer polygon
                    # polygon = polygon.buffer(-0.0001)
                    segmentation.add_polygon(polygon)
            self.segmentations.add_segmentation(key, segmentation)
            
        self.hyuclusters = data.get("nodes")
        
        # make clusterings
        # print('Getting pixel based Clustering...')
        self.clustering_pixel = pixel_based_clusterings(self.segmentations, self.path)
        
        # print('Getting node based Clustering...')
        self.clustering_nodes = nodes_clustering(self)

            
    def __repr__(self):
        return pprint.pformat(self.__dict__, indent=4)
    
    def calculate_pairwise_agreement(self, verbose = False):
        """
        Evaluate the clustering results, by pairwise agreement.
        """
        # print('Evaluate all...')
        bcubed_result = {}
        
        # pixel-based clustering
        size_names = ['size_edges_fine', 'size_edges_coarse', 'size_pixel']
        for size in size_names:
            f1, max = evaluate(self.clustering_pixel['clusters'], self.clustering_pixel['membership'], size, self.path, verbose = verbose)
            bcubed_result[size[5:]] = {'fb3': f1, 'max': max}
        # node-based clustering
        
        size_names = ['size_nodes', 'size_chars']
        for size in size_names:
            f1, max= evaluate(self.clustering_nodes['clusters'], self.clustering_nodes['membership'], size,  self.path, verbose = verbose)
            bcubed_result[size[5:]] = {'fb3': f1, 'max': max}
        
        df_bcube = pd.DataFrame(bcubed_result)
        df_bcube.to_csv(self.path + 'bcubed_res.csv')
        return df_bcube
    
    def calculate_prediction_score(self, verbose = False):
        """
        Evaluate the prediction score results, prediction has to be first key in segmentations data.
        """
        # Check data for correct form
        # segmentations
        keys = list(self.segmentations.segmentations.keys())
        if keys[0] != 'predicted':
            raise ValueError('First key in segmentations data has to be predicted.')
        if keys[1] != 'ground_truth':
            raise ValueError('Second key in segmentations data has to be ground_truth.')
        
        # nodes
        keys = list(self.hyuclusters.keys())
        if keys[0] != 'predicted':
            raise ValueError('First key in nodes data has to be predicted.')
        if keys[1] != 'ground_truth':
            raise ValueError('Second key in nodes data has to be ground_truth.')
     
        # calculate bcubed
        bcubed_result = {}
        
        # pixel-based clustering
        size_names = ['size_edges_fine', 'size_edges_coarse', 'size_pixel']
        if verbose:
            print(f'extended Bcubed results for {self.path}:')
        for size in size_names:
            f1, prec, rec = evaluate_prediction(self.clustering_pixel['clusters'], self.clustering_pixel['membership'], size, self.path, verbose = verbose)
            bcubed_result[size[5:]] = {'fb3': f1, 'precision': prec, 'recall': rec}
        
        # node-based clustering
        size_names = ['size_nodes', 'size_chars']
        for size in size_names:
            f1, prec, rec= evaluate_prediction(self.clustering_nodes['clusters'], self.clustering_nodes['membership'], size,  self.path, verbose = verbose)
            bcubed_result[size[5:]] = {'fb3': f1, 'precision': prec, 'recall': rec}
        
        df_bcube = pd.DataFrame(bcubed_result)
        df_bcube.to_csv(self.path + 'bcubed_prediction_res.csv')
        return df_bcube


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