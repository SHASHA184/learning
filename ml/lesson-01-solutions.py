"""
Solution file for Lesson 1: Vector Fundamentals
Complete implementations of all exercises from the lesson notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


class VectorCalculator:
    """A comprehensive vector calculator for ML operations"""

    @staticmethod
    def add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Add two vectors"""
        return v1 + v2

    @staticmethod
    def subtract(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Subtract second vector from first"""
        return v1 - v2

    @staticmethod
    def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate dot product"""
        return np.dot(v1, v2)

    @staticmethod
    def magnitude(v: np.ndarray) -> float:
        """Calculate vector magnitude"""
        return np.linalg.norm(v)

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit vector"""
        magnitude = np.linalg.norm(v)
        if magnitude == 0:
            return v
        return v / magnitude

    @staticmethod
    def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # Clamp to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    magnitude_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if magnitude_product == 0:
        return 0.0
    return dot_product / magnitude_product


def find_similar_users(target_user: str, users_dict: Dict[str, np.ndarray], top_k: int = 2) -> List[Tuple[str, float]]:
    """Find most similar users based on cosine similarity"""
    target_vector = users_dict[target_user]
    similarities = []

    for user, vector in users_dict.items():
        if user != target_user:
            similarity = cosine_similarity(target_vector, vector)
            similarities.append((user, similarity))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def recommend_movies(target_user: str, users_dict: Dict[str, np.ndarray],
                    genre_names: List[str], top_k: int = 2) -> Dict[str, float]:
    """Recommend genres based on similar users' preferences"""
    similar_users = find_similar_users(target_user, users_dict, top_k)
    target_preferences = users_dict[target_user]

    recommendations = {}

    for genre_idx, genre in enumerate(genre_names):
        # Calculate weighted average recommendation based on similar users
        weighted_score = 0
        total_weight = 0

        for similar_user, similarity in similar_users:
            similar_preferences = users_dict[similar_user]
            weighted_score += similarity * similar_preferences[genre_idx]
            total_weight += similarity

        if total_weight > 0:
            avg_recommendation = weighted_score / total_weight
            # Only recommend if current user hasn't rated highly
            if target_preferences[genre_idx] < avg_recommendation:
                recommendations[genre] = avg_recommendation

    return recommendations


def standardize_features(data: np.ndarray) -> np.ndarray:
    """Standardize features to have mean=0 and std=1"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    return (data - mean) / std


def normalize_features(data: np.ndarray) -> np.ndarray:
    """Normalize features to range [0, 1]"""
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)
    return (data - min_vals) / range_vals


def calculate_feature_correlations(data: np.ndarray) -> np.ndarray:
    """Calculate correlation matrix between features"""
    return np.corrcoef(data.T)


def find_k_similar_vectors(query_vector: np.ndarray, dataset: np.ndarray,
                          k: int = 3, similarity_measure: str = 'cosine') -> List[Tuple[int, float]]:
    """
    Find k most similar vectors to query vector from dataset

    Args:
        query_vector: Vector to find similarities for
        dataset: Array where each row is a data point
        k: Number of similar vectors to return
        similarity_measure: 'cosine', 'euclidean', or 'manhattan'

    Returns:
        List of tuples (index, similarity_score)
    """
    similarities = []

    for i, data_vector in enumerate(dataset):
        if similarity_measure == 'cosine':
            similarity = cosine_similarity(query_vector, data_vector)
        elif similarity_measure == 'euclidean':
            # Convert distance to similarity (lower distance = higher similarity)
            distance = np.linalg.norm(query_vector - data_vector)
            similarity = 1 / (1 + distance)  # Similarity between 0 and 1
        elif similarity_measure == 'manhattan':
            # Convert Manhattan distance to similarity
            distance = np.sum(np.abs(query_vector - data_vector))
            similarity = 1 / (1 + distance)
        else:
            raise ValueError("similarity_measure must be 'cosine', 'euclidean', or 'manhattan'")

        similarities.append((i, similarity))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:k]


def demonstrate_vector_operations():
    """Demonstrate all vector operations with examples"""
    print("=== Vector Calculator Demonstration ===")

    # Test vectors
    v1 = np.array([3, 4, 0])
    v2 = np.array([1, 0, 2])

    calc = VectorCalculator()

    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print()

    print(f"Addition: {calc.add(v1, v2)}")
    print(f"Subtraction: {calc.subtract(v1, v2)}")
    print(f"Dot product: {calc.dot_product(v1, v2)}")
    print(f"Magnitude of v1: {calc.magnitude(v1):.3f}")
    print(f"Magnitude of v2: {calc.magnitude(v2):.3f}")
    print(f"Normalized v1: {calc.normalize(v1)}")
    print(f"Angle between vectors: {calc.angle_between(v1, v2):.3f} radians")
    print(f"Cosine similarity: {cosine_similarity(v1, v2):.3f}")


def demonstrate_recommendation_system():
    """Demonstrate recommendation system with movie ratings"""
    print("\n=== Recommendation System Demonstration ===")

    # Movie rating data
    users = {
        'Alice': np.array([5, 2, 4, 1, 3]),  # [Action, Comedy, Drama, Horror, Romance]
        'Bob': np.array([2, 5, 2, 1, 4]),
        'Carol': np.array([4, 3, 5, 2, 2]),
        'David': np.array([1, 4, 2, 5, 1]),
        'Eve': np.array([5, 1, 3, 1, 4])
    }

    genre_names = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']

    target_user = 'Alice'
    similar_users = find_similar_users(target_user, users)

    print(f"Users most similar to {target_user}:")
    for user, similarity in similar_users:
        print(f"  {user}: {similarity:.3f}")

    recommendations = recommend_movies(target_user, users, genre_names)
    print(f"\nRecommendations for {target_user}:")
    for genre, score in sorted(recommendations.items(), key=lambda x: x[1], reverse=True):
        print(f"  {genre}: {score:.2f}")


def demonstrate_data_preprocessing():
    """Demonstrate data preprocessing techniques"""
    print("\n=== Data Preprocessing Demonstration ===")

    # House data: [size_sqft, bedrooms, bathrooms, age_years, price]
    house_data = np.array([
        [2000, 3, 2, 10, 300000],
        [1500, 2, 1, 20, 250000],
        [3000, 4, 3, 5, 500000],
        [1200, 2, 1, 30, 200000],
        [2500, 3, 2, 15, 400000]
    ])

    print("Original data:")
    print(house_data)

    print("\nStandardized data (mean=0, std=1):")
    standardized = standardize_features(house_data)
    print(standardized)
    print(f"Means: {np.mean(standardized, axis=0)}")
    print(f"Stds: {np.std(standardized, axis=0)}")

    print("\nNormalized data (range [0,1]):")
    normalized = normalize_features(house_data)
    print(normalized)
    print(f"Mins: {np.min(normalized, axis=0)}")
    print(f"Maxs: {np.max(normalized, axis=0)}")

    print("\nFeature correlations:")
    correlations = calculate_feature_correlations(house_data)
    feature_names = ['Size', 'Bedrooms', 'Bathrooms', 'Age', 'Price']
    print("Correlation matrix:")
    for i, name in enumerate(feature_names):
        print(f"{name:10}", end="")
        for j in range(len(feature_names)):
            print(f"{correlations[i,j]:8.3f}", end="")
        print()


def demonstrate_similarity_search():
    """Demonstrate k-similar vector search"""
    print("\n=== Similarity Search Demonstration ===")

    # Customer data: [age, income, spending_score, loyalty_years]
    customers = np.array([
        [25, 50000, 80, 2],
        [45, 80000, 60, 5],
        [35, 60000, 70, 3],
        [28, 45000, 85, 1],
        [50, 90000, 40, 8],
        [32, 55000, 75, 4]
    ])

    query_customer = np.array([30, 55000, 75, 4])

    print(f"Query customer: {query_customer}")
    print("Customer database:")
    for i, customer in enumerate(customers):
        print(f"  Customer {i}: {customer}")

    # Find similar customers using different measures
    for measure in ['cosine', 'euclidean', 'manhattan']:
        print(f"\nTop 3 similar customers using {measure} similarity:")
        similar_customers = find_k_similar_vectors(query_customer, customers, k=3, similarity_measure=measure)
        for idx, similarity in similar_customers:
            print(f"  Customer {idx}: {customers[idx]} (similarity: {similarity:.3f})")


if __name__ == "__main__":
    demonstrate_vector_operations()
    demonstrate_recommendation_system()
    demonstrate_data_preprocessing()
    demonstrate_similarity_search()

    print("\n=== Quiz Answers ===")
    print("1. Cosine similarity between [3,4] and [6,8]:")
    v1, v2 = np.array([3, 4]), np.array([6, 8])
    print(f"   Answer: {cosine_similarity(v1, v2):.3f}")

    print("2. Normalized random vector of 100 elements:")
    random_vec = np.random.random(100)
    normalized_random = VectorCalculator.normalize(random_vec)
    print(f"   Magnitude of normalized vector: {VectorCalculator.magnitude(normalized_random):.6f}")

    print("3. Closest customer to [30, 55000, 75, 4]:")
    # This was demonstrated in the similarity search above