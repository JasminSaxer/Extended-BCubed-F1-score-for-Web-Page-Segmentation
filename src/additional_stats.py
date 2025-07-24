from lxml import etree
from src.pixel_clustering import as_geometric_collection, explain_validity, split_polygons, size_function, apply_canny


def get_detailed_stats_per_class(clustering, memberships, atomic_element, total_size_of_ground_truth, none_predicted, none_groundtruth, verbose=False):
    # not in spirit of bcubed
    # predicted = 0, ground truth = 1 (index)
    
    # go trough all the predicted segments (=clusters)
    # get the index of the pred/ground truth in the original_clusters list (important for split clusters memberships)
    
    # if none_predicted or none_groundtruth set list to empty
    if none_predicted:
        index_of_segments_predicted = []
        index_of_segments_ground_truth = memberships[0]
        
    elif none_groundtruth:
        index_of_segments_predicted = memberships[0]
        index_of_segments_ground_truth = []
    else:
        index_of_segments_predicted = memberships[0]
        index_of_segments_ground_truth = memberships[1]
        
    # all ground truth split cluster indexes
    groundtruth_split_clusters_all = get_splitclusters_from_orginal_clusters(index_of_segments_ground_truth, clustering)
    predicted_split_clusters_all = get_splitclusters_from_orginal_clusters(index_of_segments_predicted, clustering) 

    # Calculate the overlap between predicted and ground truth split clusters
    # TP_splitclusters = set(predicted_split_clusters_all) & set(groundtruth_split_clusters_all)
    FP_splitclusters = set(predicted_split_clusters_all) - set(groundtruth_split_clusters_all)
    FN_splitclusters = set(groundtruth_split_clusters_all) - set(predicted_split_clusters_all)
    
    # TP_size = get_size_of_cluster(size_name, TP_splitclusters, clustering)
    FP_size = get_size_of_cluster(atomic_element, FP_splitclusters, clustering)
    FN_size = get_size_of_cluster(atomic_element, FN_splitclusters, clustering)
    
    # get relative sizes to total ground truth
    FP_size_rel = FP_size / total_size_of_ground_truth[atomic_element]
    FN_size_rel = FN_size / total_size_of_ground_truth[atomic_element]
        
    return {
        'FP_size_rel': FP_size_rel,
        'FN_size_rel': FN_size_rel}

    
def get_splitclusters_from_orginal_clusters(original_clusters_idx, clustering):
    split_clusters_idx = []
    
    if isinstance(clustering, list):
        clusters = clustering
    else:
        clusters = clustering.clusters
        
    for i in original_clusters_idx:
        for j, split_cluster in enumerate(clusters):
            if split_cluster['membership'][i] == 1:
                split_clusters_idx.append(j)
                
    return split_clusters_idx
        
    
def get_size_of_cluster(size_name, split_cluster_idx, clustering):
    size = 0
    
    if isinstance(clustering, list):
        clusters = clustering
    else:
        clusters = clustering.clusters
        
        
    for idx in split_cluster_idx:
        size = clusters[idx][f'size_{size_name}']
        size += size
    return size


def get_size_of_total_groundtruth_from_clusters(clusters, atomic_elements, task ):

    sizes = {}
    if 'nodes' in atomic_elements:
        clusters_nb = clusters.get('node_based', {})
        if 'ground_truth' in clusters_nb:
     
            gt_clusters = clusters_nb['ground_truth']
            
            hyu_list = [c['hyuIndex'] for c in gt_clusters]
            
            with open(task.path_to_html, 'r') as html_file:
                html = html_file.read()
            
            dom = etree.HTML(html)
            elements = get_elements_by_hyu(dom, hyu_list)
            # check that not counting nested elements if 
            # they are overlapping
            non_nested = filter_non_overlapping(elements)
            # sizes['chars']  = sum(count_text_length(el) for el in non_nested)
            sizes['nodes'] = sum(count_dom_nodes(el) for el in non_nested)          
            sizes['chars'] = sum(count_text_length(el) for el in non_nested)

    if 'pixel' in atomic_elements:
        clusters_pb = clusters.get('pixel_based', {})
        if 'ground_truth' in clusters_pb:
            gt_clusters = clusters_pb['ground_truth']
                
            task.transform_to_shapely_polygons({'ground_truth':gt_clusters})
            segmentations = task.segmentations
            # Convert segmentations to a GeometryCollection        
            segmentations.geometrics = as_geometric_collection(segmentations) 
            geom = segmentations.geometrics
            if not geom.is_valid:
                print(f"Invalid geometry: {explain_validity(geom)}")

                geom = geom.buffer(0)  # Attempt to fix the geometry
                
                if not geom.is_valid:
                    raise ValueError(f"Could not fix geometry: {explain_validity(geom)}")
                    
            # Split the overlapping polygons into distinct segments
            distinct_segment = split_polygons(segmentations.geometrics.geoms)
            
            # get edges with canny
            edges_fine = apply_canny(task.path_to_screenshot, 'fine')
            edges_coarse = apply_canny(task.path_to_screenshot, 'coarse')

            sizes['pixel'] = sum(size_function(seg, 'pixel', None) for seg in distinct_segment)
            sizes['edges_fine'] = sum(size_function(seg, 'edges', edges_fine) for seg in distinct_segment)
            sizes['edges_coarse'] = sum(size_function(seg, 'edges', edges_coarse) for seg in distinct_segment)

    return sizes
    
    
    

def get_elements_by_hyu(tree, hyu_list):
    hyu_set = set(map(str, hyu_list))
    return [el for el in tree.xpath('//*[@hyu]') if el.get('hyu') in hyu_set]

def filter_non_overlapping(elements):
    non_nested = []
    for el in elements:
        if not any(other is not el and _is_ancestor(other, el) for other in elements):
            non_nested.append(el)
    return non_nested

def _is_ancestor(ancestor, element):
    parent = element.getparent()
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.getparent()
    return False

def count_text_length(el):
    return len(''.join(el.itertext()))

def count_dom_nodes(el):
    return sum(1 for _ in el.iter())  # includes self
