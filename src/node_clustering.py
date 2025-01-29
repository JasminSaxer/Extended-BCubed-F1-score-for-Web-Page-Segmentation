from lxml import etree
import numpy as np

def nodes_clustering(task):
    with open(f"{task.path}dom.html", 'r') as html_file:
        html = html_file.read()
    dom = etree.HTML(html)
    
    # get all nodes per 'cluster'
    res = {}
    text_nodes = {}
    for user in task.hyuclusters:
        res[user] = []
        clusters = task.hyuclusters[user]
        for cluster in clusters:
            node = dom.xpath(f"//*[@hyu='{cluster['hyuIndex']}']")
            node_hyu = int(cluster['hyuIndex'])
            # get text length of this node
            if node[0].text and node[0].text.strip():
                text_nodes[node_hyu] = len(node[0].text.strip())
            
            # start hyu values with this node    
            hyu_values = set([node_hyu])
            
            # get all children nodes
            for child in node[0].iterdescendants():
                hyu_child = child.get('hyu')
                if hyu_child:
                    hyu_values.add(int(hyu_child))
                    # get text length of the children node
                    if child.text and child.text.strip():
                        text_nodes[int(hyu_child)] = len(child.text.strip())
  
            res[user].append(hyu_values)


    # Remove duplicate sets from unique_clusters
    unique_clusters = [cluster for user in res for cluster in res[user]]
    unique_clusters = list(set([frozenset(cluster) for cluster in unique_clusters]))

    split_clusters = split_and_list_clusters(unique_clusters.copy())
    
    # Build membership matrix
    membership = {}
    membership['matrix'] = np.zeros((len(split_clusters), len(unique_clusters)), dtype=int)

    for i, distinct_poly in enumerate(split_clusters):
        for j, original_poly in enumerate(unique_clusters):

            intersects = distinct_poly.intersection(original_poly)
            if len(intersects) > 0:
                membership['matrix'][i, j] = 1


    # Define subsets for each annotator
    membership_subsets = get_subsets(res, unique_clusters)  # Placeholder for subset logic
    clusterings = {'clusters': [], 
                   'membership': membership_subsets}
    
    # get edges if canny is used
    for i, cluster in enumerate(split_clusters):
        info = {
            'membership': membership['matrix'][i],
            'size_nodes': len(cluster),
            'size_chars': get_size(cluster, text_nodes)
        }
        clusterings['clusters'].append(info)
    return clusterings
    

def get_size(cluster, text_nodes):
    size = 0
    for node in cluster:
        size += text_nodes.get(node, 0)
    return size
    
    
def get_subsets(res, unique_clusters):
    dict_segmens = {c:i for i, c in enumerate(unique_clusters)}
    subsets = []
    for user, clusters in res.items():
        subsets_annot = []
        for cluster in clusters:
            subsets_annot.append(dict_segmens[frozenset(cluster)])
        subsets.append(subsets_annot)
    return subsets


def split_and_list_clusters(clusters):
    """
    Splits overlapping clusters into non-overlapping clusters.

    :param clusters: List of clusters (each cluster is a set of numbers)
    :return: A list of non-overlapping clusters
    """
    non_overlapping_clusters = []

    while clusters:
        # Start with the first cluster and remove it
        current = clusters.pop(0)
        new_clusters = []

        for other in clusters:
            # Compute overlap and differences
            overlap = current & other
            diff_current = current - other
            diff_other = other - current

            # Update the current cluster with non-overlapping part
            if overlap:
                new_clusters.append(overlap)
            if diff_other:
                new_clusters.append(diff_other)
            current = diff_current

        # Add the non-overlapping part of the current cluster
        if current:
            non_overlapping_clusters.append(current)

        # Update the clusters list for the next iteration
        clusters = new_clusters

    return non_overlapping_clusters
