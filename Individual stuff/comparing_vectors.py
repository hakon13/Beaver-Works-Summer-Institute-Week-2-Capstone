import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from database import data, get_img_vector

def compare_face():
    query_vector = get_img_vector()
    compared_database = {}

    if query_vector is not None:
        query_vector = np.asarray(query_vector).reshape(1, -1)

        for name, profile in data.items():
            if len(profile.descriptors) == 0:
                continue

            average_vector = np.mean(
                profile.descriptors,
                axis=0
            ).reshape(1, -1)

            compared_database[name] = cosine_distances(
                query_vector,
                average_vector
            )[0, 0]

    print(compared_database)

    return compared_database