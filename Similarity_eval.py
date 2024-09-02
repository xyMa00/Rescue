import sys
import numpy as np

# input 2 vector array
# output pearson correlation score
def PearsonCorrelationSimilarity(vec1, vec2):
    value = range(len(vec1))

    sum_vec1 = sum([vec1[i] for i in value])
    sum_vec2 = sum([vec2[i] for i in value])

    square_sum_vec1 = sum([pow(vec1[i], 2) for i in value])
    square_sum_vec2 = sum([pow(vec2[i], 2) for i in value])

    product = sum([vec1[i] * vec2[i] for i in value])

    numerator = product - (sum_vec1 * sum_vec2 / len(vec1))
    dominator = ((square_sum_vec1 - pow(sum_vec1, 2) / len(vec1)) * (
                square_sum_vec2 - pow(sum_vec2, 2) / len(vec2))) ** 0.5

    if dominator == 0:
        return 0
    result = numerator / (dominator * 1.0)

    return result

def dice_coefficient(a, b):
    """dice coefficient 2nt/na + nb."""
    # a_bigrams = set(a)
    # b_bigrams = set(b)
    # overlap = len(a_bigrams & b_bigrams)
    # # print("overlap", overlap)
    # return overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    sum1 = 0
    count_all = 0
    for a1, b1 in zip(a, b):
        sum1 += a1 * b1
        count_all += pow(a1, 2) + pow(b1, 2)
    dice = 2*sum1/count_all

    return dice


# def squared_chord_similarity(list1, list2):
#     # Convert lists to NumPy arrays
#     vec1 = np.array(list1)
#     vec2 = np.array(list2)
#     # Compute the squared-chord similarity
#     similarity = np.sum(np.sqrt(vec1 * vec2))
#     return similarity

def squared_chord_similarity(a, b):
    dis = 0
    for a1, b1 in zip(a, b):
        dife = pow(a1, 1/2) - pow(b1, 1/2)
        mid = pow(dife, 2)
        dis += mid
    squared_chord = 1-dis
    return squared_chord


# def jaccard_similarity(list1, list2):
#     # Convert lists to sets
#     set1 = set(list1)
#     set2 = set(list2)
#     # Compute intersection and union
#     intersection = set1.intersection(set2)
#     union = set1.union(set2)
#     # Compute Jaccard Similarity
#     similarity = len(intersection) / len(union)
#
#     return similarity

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    # print("len(s1.intersection(s2)):",len(s1.intersection(s2)))
    # print("len(s1.union(s2)):", len(s1.union(s2)))
    return len(s1.intersection(s2)) / len(s1.union(s2))


def fidelity_similarity(list1, list2):
    # Ensure the lists are numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Normalize the arrays to make them probability distributions
    arr1 = arr1 / np.sum(arr1)
    arr2 = arr2 / np.sum(arr2)

    # Compute the Fidelity similarity (Bhattacharyya coefficient)
    bc = np.sum(np.sqrt(arr1 * arr2))

    return bc


def euclidean_distance_similarity(list1, list2):
    # Convert lists to numpy arrays
    arr1 = np.array(list1)
    arr2 = np.array(list2)

    # Compute the Euclidean distance
    distance = np.linalg.norm(arr1 - arr2)

    # Convert distance to similarity score
    # A common approach is to use a negative exponential function to convert the distance to a similarity score
    similarity = np.exp(-distance)

    return similarity



def python_cos(q_vec, b_vec):
    """
    计算余弦相似度
    :param q_vec: 一维数组
    :param b_vec: 一维数组
    :return:
    """
    dot_q_b = 0
    q_vec_length = 0
    b_vec_length = 0
    for q, b in zip(q_vec, b_vec):
        dot_q_b += q * b
        q_vec_length += q * q
        b_vec_length += b * b
    length = (q_vec_length ** (1 / 2)) * (b_vec_length ** (1 / 2))
    cos_sim = dot_q_b / length #向量的内积除以向量模长的积
    # print('cos_sim',cos_sim)
    return cos_sim

# if __name__ == "__main__":
#     # vec1 = [5.0, 3.0, 2.5]
#     # vec2 = [2.0, 2.5, 5.0]
#
#
#     PearsonCorrelationSimilarity(vec1, vec2)