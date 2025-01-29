import numpy as np
from pprint import pprint

class Clustering:
    def __init__(self, clusters, membership_subsets, path):
        self.clusters = clusters
        self.membership_subsets = membership_subsets
        self.path = path

def AgreementMatrices(clustering):
    agreement_matrices = [AgreementMatrix(clustering, membership_subset) for membership_subset in clustering.membership_subsets]
    
    return agreement_matrices

def AgreementMatrix(clustering, membership_subset):
    def GetAgreement(cluster_indices):
        cluster1_membership = clustering.clusters[cluster_indices[0]]['membership'][membership_subset]
        cluster2_membership = clustering.clusters[cluster_indices[1]]['membership'][membership_subset]
        return np.sum(cluster1_membership & cluster2_membership)

    num_clusters = len(clustering.clusters)
    common_segment_counts = np.full((num_clusters, num_clusters), np.nan)

    # Fill lower triangle
    for i in range(num_clusters):
        for j in range(i + 1):
            common_segment_counts[i, j] = GetAgreement([i, j])

    # Make symmetric
    upper_triangle_indices = np.triu_indices(num_clusters, k=1)
    common_segment_counts[upper_triangle_indices] = common_segment_counts.T[upper_triangle_indices]
    
    return common_segment_counts

def BCubedPrecisionMatrix(clustering, size_name, agreement_matrices=None ):
    if agreement_matrices is None:
        agreement_matrices = AgreementMatrices(clustering)
    

    cluster_sizes = np.array([cluster[size_name] for cluster in clustering.clusters])
    

    num_clusters = len(clustering.clusters)

    def MultiplicityPrecision(l1, l2, agreement_matrix_algorithm, agreement_matrix_truth):
        num_segments_algorithm = agreement_matrix_algorithm[l1, l2]
        num_segments_truth = agreement_matrix_truth[l1, l2]
        if num_segments_algorithm == 0:
            return np.nan
        else:
            res = min(num_segments_algorithm, num_segments_truth) / num_segments_algorithm            
            return res

    def AverageMultiplicityPrecision(l1, agreement_matrix_algorithm, agreement_matrix_truth):
        multiplicity_precision = np.array([MultiplicityPrecision(l1, l2, agreement_matrix_algorithm, agreement_matrix_truth) for l2 in range(num_clusters)])
        common = ~np.isnan(multiplicity_precision)
        total = np.sum(cluster_sizes[common])
        if total == 0:
            return np.nan
        else:
            return np.sum(multiplicity_precision[common] * cluster_sizes[common]) / total

    def BCubedPrecision(agreement_matrix_algorithm_index, agreement_matrix_truth_index):
        if agreement_matrix_algorithm_index == agreement_matrix_truth_index:
            return 1.0

        average_multiplicity_precision = np.array([AverageMultiplicityPrecision(l1, agreement_matrices[agreement_matrix_algorithm_index], agreement_matrices[agreement_matrix_truth_index]) for l1 in range(num_clusters)])
        common = ~np.isnan(average_multiplicity_precision)
        total = np.sum(cluster_sizes[common])
        if total == 0:
            return 0
        bcubed_precision = np.sum(average_multiplicity_precision[common] * cluster_sizes[common]) / total
        return bcubed_precision

    num_matrices = len(agreement_matrices)
    bcubed_precision_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in range(num_matrices):
        for j in range(num_matrices):
            bcubed_precision_matrix[i, j] = BCubedPrecision(i, j)

    return bcubed_precision_matrix


def CombineBCubedPrecisionAndRecall(bcubed_precision_matrix, combination_function):
    # Create a copy of the precision matrix to store the combined values
    bcubed_combined_matrix = bcubed_precision_matrix.copy()
    
    # Get the lower triangle of the matrix (excluding the diagonal)
    lower_triangle_indices = np.tril_indices_from(bcubed_combined_matrix, k=-1)
    
    # Apply the combination function to each pair (precision, recall)
    for i, j in zip(*lower_triangle_indices):
        recall = bcubed_combined_matrix[i, j]
        precision = bcubed_combined_matrix[j, i]
        bcubed_combined_matrix[i, j] = combination_function(precision, recall)
    
    # Make the matrix symmetric by copying the lower triangle to the upper triangle
    upper_triangle_indices = np.triu_indices_from(bcubed_combined_matrix, k=1)
    bcubed_combined_matrix[upper_triangle_indices] = bcubed_combined_matrix.T[upper_triangle_indices]
    
    return bcubed_combined_matrix


def BCubedF1Matrix(bcubed_precision_matrix):
    # Check if input is a BCubedPrecisionMatrix (a subclass of matrix)
    if not isinstance(bcubed_precision_matrix, np.ndarray):
        raise TypeError(f"Method requires a BCubedPrecisionMatrix, but got a {type(bcubed_precision_matrix)}")

    # Combine precision and recall to compute F1 using the given CombineBCubedPrecisionAndRecall function
    bcubed_f1_matrix = CombineBCubedPrecisionAndRecall(bcubed_precision_matrix, F1)
    # If you need to add a specific attribute for "BCubedF1Matrix" class, you can do it here
    return bcubed_f1_matrix

def BCubedMaxMatrix(bcubed_precision_matrix):
    # Check if input is a BCubedPrecisionMatrix
    if not isinstance(bcubed_precision_matrix, np.ndarray):
        raise TypeError(f"Method requires a BCubedPrecisionMatrix, but got a {type(bcubed_precision_matrix)}")

    # Combine precision and recall to compute the maximum
    bcubed_max_matrix = CombineBCubedPrecisionAndRecall(bcubed_precision_matrix, np.maximum)
    # Add any necessary attributes if needed
    return bcubed_max_matrix

def EvaluationMatrixMean(evaluation_matrix):
    # Check if input is an EvaluationMatrix (a subclass of matrix)
    if not isinstance(evaluation_matrix, np.ndarray):
        raise TypeError(f"Method requires an EvaluationMatrix, but got a {type(evaluation_matrix)}")

    # Compute the mean of the lower triangular part of the matrix (excluding diagonal)
    lower_triangle = lower_triangle_with_nan(evaluation_matrix)  # lower triangle, excluding diagonal
    return  np.nanmean(lower_triangle)# Use nanmean to ignore NaNs if they are present

def F1(precision, recall):
    if not (precision and recall):
        return 0
    return 2 * precision * recall / (precision + recall)

def lower_triangle_with_nan(matrix):
    # Create a matrix of the same shape, filled with NaNs
    result_matrix = np.full_like(matrix, np.nan, dtype=float)
    # Set the lower triangle of the result matrix to be the lower triangle of the input matrix
    lower_triangle_indices = np.tril_indices_from(matrix, k=-1)  # k=0 to include the diagonal
    result_matrix[lower_triangle_indices] = matrix[lower_triangle_indices]
    return result_matrix

def evaluate(clusterings, memberships, size_name, path, verbose=False):
    clustering = Clustering(clusterings, memberships, path)

    agreement_matrices = AgreementMatrices(clustering)
    # pprint(agreement_matrices)
    bcubed_precision_matrix = BCubedPrecisionMatrix(clustering, size_name, agreement_matrices)

    f1_mean = EvaluationMatrixMean(BCubedF1Matrix(bcubed_precision_matrix))
    max_mean = EvaluationMatrixMean(BCubedMaxMatrix(bcubed_precision_matrix))
    if verbose:
        print(size_name, 'BCubed Precision Matrix:')
        pprint(bcubed_precision_matrix)
        print('F1', f1_mean)
        print('Max', max_mean)
        
    return f1_mean, max_mean

def evaluate_prediction(clusterings, memberships, size_name, path, verbose=False):
    clustering = Clustering(clusterings, memberships, path)

    agreement_matrices = AgreementMatrices(clustering)
    bcubed_precision_matrix = BCubedPrecisionMatrix(clustering, size_name, agreement_matrices)

    precision = bcubed_precision_matrix[0][1]
    recall = bcubed_precision_matrix[1][0]
    
    f1 = F1(precision, recall)

    if verbose:
        print(size_name[5:], '\tF1:', round(f1,3), ', Precision:', round(precision,3), ', Recall:', round(recall,3))

    return f1, precision, recall
