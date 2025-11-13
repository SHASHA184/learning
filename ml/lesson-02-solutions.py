"""
Solution file for Lesson 2: Matrix Operations
Complete implementations of all exercises from the lesson notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore', category=np.RankWarning)


class MatrixCalculator:
    """A comprehensive matrix calculator for ML operations"""

    @staticmethod
    def multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication with dimension checking"""
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Cannot multiply matrices: {A.shape} and {B.shape}. "
                           f"Inner dimensions must match: {A.shape[1]} != {B.shape[0]}")
        return A @ B

    @staticmethod
    def transpose(A: np.ndarray) -> np.ndarray:
        """Matrix transpose"""
        return A.T

    @staticmethod
    def inverse(A: np.ndarray) -> np.ndarray:
        """Matrix inverse with singularity checking"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square for inversion. Got shape: {A.shape}")

        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            raise np.linalg.LinAlgError("Matrix is singular (determinant ≈ 0) and cannot be inverted")

        return np.linalg.inv(A)

    @staticmethod
    def determinant(A: np.ndarray) -> float:
        """Calculate determinant"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square for determinant. Got shape: {A.shape}")
        return np.linalg.det(A)

    @staticmethod
    def eigenvalues(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors"""
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square for eigendecomposition. Got shape: {A.shape}")

        eigenvals, eigenvecs = np.linalg.eig(A)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        return eigenvals, eigenvecs

    @staticmethod
    def svd(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition"""
        return np.linalg.svd(A)

    @staticmethod
    def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b"""
        if A.shape[0] != A.shape[1]:
            # For overdetermined systems, use least squares
            return np.linalg.lstsq(A, b, rcond=None)[0]

        det = np.linalg.det(A)
        if abs(det) < 1e-10:
            # Use pseudoinverse for singular matrices
            return np.linalg.pinv(A) @ b

        return np.linalg.solve(A, b)


def create_synthetic_image(size: int = 50) -> np.ndarray:
    """Create a synthetic image for processing"""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    # Create a Gaussian-like pattern
    image = np.exp(-(x**2 + y**2))
    # Add some patterns
    image += 0.3 * np.sin(5 * x) * np.cos(5 * y)
    # Add noise
    noise = 0.1 * np.random.randn(size, size)
    return np.clip(image + noise, 0, 1)


def apply_filter(image: np.ndarray, filter_kernel: np.ndarray) -> np.ndarray:
    """Apply a filter to an image using convolution (simplified)"""
    if len(filter_kernel.shape) != 2 or filter_kernel.shape[0] != filter_kernel.shape[1]:
        raise ValueError("Filter kernel must be a square 2D array")

    if filter_kernel.shape[0] % 2 == 0:
        raise ValueError("Filter kernel size must be odd")

    kernel_size = filter_kernel.shape[0]
    pad_size = kernel_size // 2

    # Pad the image
    padded_image = np.pad(image, pad_size, mode='edge')
    filtered_image = np.zeros_like(image)

    # Apply convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract neighborhood
            neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Apply filter
            filtered_image[i, j] = np.sum(neighborhood * filter_kernel)

    return filtered_image


def compress_image_svd(image: np.ndarray, n_components: int) -> Tuple[np.ndarray, float]:
    """Compress image using SVD"""
    U, s, Vt = np.linalg.svd(image)

    # Keep only top n_components
    U_reduced = U[:, :n_components]
    s_reduced = s[:n_components]
    Vt_reduced = Vt[:n_components, :]

    # Reconstruct image
    compressed = U_reduced @ np.diag(s_reduced) @ Vt_reduced

    # Calculate compression ratio
    original_size = image.size
    compressed_size = U_reduced.size + s_reduced.size + Vt_reduced.size
    compression_ratio = original_size / compressed_size

    return compressed, compression_ratio


class NeuralNetworkLayer:
    """A simple neural network layer using matrix operations"""

    def __init__(self, input_size: int, output_size: int, random_seed: int = 42):
        np.random.seed(random_seed)
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights with small random values (Xavier initialization)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        # Initialize biases to zeros
        self.biases = np.zeros(output_size)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: X @ W + b"""
        if X.shape[1] != self.input_size:
            raise ValueError(f"Input size mismatch: expected {self.input_size}, got {X.shape[1]}")

        # Linear transformation
        Z = X @ self.weights + self.biases
        return Z

    def apply_activation(self, Z: np.ndarray, activation: str = 'relu') -> np.ndarray:
        """Apply activation function"""
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'sigmoid':
            # Clip to prevent overflow
            Z_clipped = np.clip(Z, -500, 500)
            return 1 / (1 + np.exp(-Z_clipped))
        elif activation == 'tanh':
            return np.tanh(Z)
        elif activation == 'linear':
            return Z
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def get_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return weights and biases"""
        return self.weights.copy(), self.biases.copy()

    def set_parameters(self, weights: np.ndarray, biases: np.ndarray) -> None:
        """Set weights and biases"""
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights.shape}, got {weights.shape}")
        if biases.shape != self.biases.shape:
            raise ValueError(f"Bias shape mismatch: expected {self.biases.shape}, got {biases.shape}")

        self.weights = weights.copy()
        self.biases = biases.copy()


class DataTransformer:
    """Data preprocessing pipeline using matrix operations"""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.pca_components_ = None
        self.pca_mean_ = None
        self.pca_explained_variance_ = None
        self.is_fitted_standardize = False
        self.is_fitted_pca = False

    def standardize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Standardize features to have mean=0, std=1"""
        if fit:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            # Avoid division by zero
            self.std_ = np.where(self.std_ == 0, 1, self.std_)
            self.is_fitted_standardize = True
        elif not self.is_fitted_standardize:
            raise ValueError("Must fit standardizer first or set fit=True")

        return (X - self.mean_) / self.std_

    def fit_pca(self, X: np.ndarray, n_components: Optional[int] = None) -> None:
        """Fit PCA transformation"""
        if not self.is_fitted_standardize:
            # Standardize first
            X_std = self.standardize(X, fit=True)
        else:
            X_std = self.standardize(X, fit=False)

        # Center the data
        self.pca_mean_ = np.mean(X_std, axis=0)
        X_centered = X_std - self.pca_mean_

        # Calculate covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        # Keep only n_components
        if n_components is None:
            n_components = len(eigenvals)

        self.pca_components_ = eigenvecs[:, :n_components]
        self.pca_explained_variance_ = eigenvals[:n_components]
        self.is_fitted_pca = True

    def transform_pca(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA transformation"""
        if not self.is_fitted_pca:
            raise ValueError("Must fit PCA first")

        # Standardize and center
        X_std = self.standardize(X, fit=False)
        X_centered = X_std - self.pca_mean_

        # Transform
        return X_centered @ self.pca_components_

    def add_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Add polynomial features"""
        if degree < 1:
            raise ValueError("Degree must be at least 1")

        n_samples, n_features = X.shape
        features = [np.ones((n_samples, 1))]  # Bias term

        # Add original features
        features.append(X)

        if degree >= 2:
            # Add squared terms
            for i in range(n_features):
                features.append(X[:, i:i+1] ** 2)

            # Add interaction terms
            for i in range(n_features):
                for j in range(i+1, n_features):
                    features.append((X[:, i] * X[:, j]).reshape(-1, 1))

        if degree >= 3:
            # Add cubic terms (simplified - just original features cubed)
            for i in range(n_features):
                features.append(X[:, i:i+1] ** 3)

        return np.concatenate(features, axis=1)

    def create_interaction_matrix(self, X: np.ndarray) -> np.ndarray:
        """Create matrix of all pairwise feature interactions"""
        n_samples, n_features = X.shape
        interactions = []

        for i in range(n_features):
            for j in range(i, n_features):  # Include diagonal (x_i * x_i)
                interaction = X[:, i] * X[:, j]
                interactions.append(interaction.reshape(-1, 1))

        return np.concatenate(interactions, axis=1)

    def get_pca_info(self) -> dict:
        """Get PCA information"""
        if not self.is_fitted_pca:
            return {}

        total_variance = np.sum(self.pca_explained_variance_)
        explained_variance_ratio = self.pca_explained_variance_ / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        return {
            'explained_variance': self.pca_explained_variance_,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'n_components': len(self.pca_explained_variance_)
        }


def demonstrate_matrix_calculator():
    """Demonstrate matrix calculator functionality"""
    print("=== Matrix Calculator Demonstration ===")

    calc = MatrixCalculator()

    # Test matrices
    A = np.array([[4, 2], [3, 1]])
    B = np.array([[1, 3], [2, 4]])
    b = np.array([10, 7])

    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Vector b: {b}")
    print()

    # Test operations
    print(f"Matrix multiplication A @ B:\n{calc.multiply(A, B)}")
    print(f"Transpose of A:\n{calc.transpose(A)}")
    print(f"Determinant of A: {calc.determinant(A):.3f}")
    print(f"Inverse of A:\n{calc.inverse(A)}")
    print(f"Linear system solution (Ax = b): {calc.solve_linear_system(A, b)}")

    eigenvals, eigenvecs = calc.eigenvalues(A)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")

    U, s, Vt = calc.svd(A)
    print(f"SVD - Singular values: {s}")
    print()


def demonstrate_image_processing():
    """Demonstrate image processing with matrices"""
    print("=== Image Processing Demonstration ===")

    # Create synthetic image
    image = create_synthetic_image(30)
    print(f"Image shape: {image.shape}")
    print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")

    # Define filters
    blur_kernel = np.ones((3, 3)) / 9
    edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Apply filters
    blurred = apply_filter(image, blur_kernel)
    edges = apply_filter(image, edge_kernel)

    # Compress using SVD
    compressed_5, ratio_5 = compress_image_svd(image, 5)
    compressed_10, ratio_10 = compress_image_svd(image, 10)

    print(f"Compression ratio (5 components): {ratio_5:.2f}x")
    print(f"Compression ratio (10 components): {ratio_10:.2f}x")

    # Visualize results
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred (Average Filter)')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.imshow(compressed_5, cmap='gray')
    plt.title(f'SVD Compressed (5 components, {ratio_5:.1f}x)')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.imshow(compressed_10, cmap='gray')
    plt.title(f'SVD Compressed (10 components, {ratio_10:.1f}x)')
    plt.colorbar()

    plt.subplot(2, 3, 6)
    # Plot compression quality vs components
    components = range(1, min(image.shape) + 1)
    mse_values = []
    for n in components:
        compressed, _ = compress_image_svd(image, n)
        mse = np.mean((image - compressed) ** 2)
        mse_values.append(mse)

    plt.plot(components, mse_values)
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error')
    plt.title('Compression Quality vs Components')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print()


def demonstrate_neural_network():
    """Demonstrate neural network layer"""
    print("=== Neural Network Layer Demonstration ===")

    # Create test data
    batch_size, input_dim, hidden_dim = 32, 10, 5
    test_input = np.random.randn(batch_size, input_dim)

    # Create layer
    layer = NeuralNetworkLayer(input_dim, hidden_dim)

    print(f"Input shape: {test_input.shape}")
    print(f"Layer: {input_dim} -> {hidden_dim}")

    # Forward pass
    output = layer.forward(test_input)
    print(f"Output shape: {output.shape}")
    print(f"Output statistics: mean={np.mean(output):.3f}, std={np.std(output):.3f}")

    # Test different activations
    activations = ['linear', 'relu', 'sigmoid', 'tanh']

    plt.figure(figsize=(15, 4))

    for i, activation in enumerate(activations):
        activated = layer.apply_activation(output, activation)

        plt.subplot(1, 4, i+1)
        plt.hist(activated.flatten(), bins=30, alpha=0.7)
        plt.title(f'{activation.capitalize()} Activation')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    weights, biases = layer.get_parameters()
    print(f"Weights shape: {weights.shape}")
    print(f"Biases shape: {biases.shape}")
    print()


def demonstrate_data_transformer():
    """Demonstrate data transformation pipeline"""
    print("=== Data Transformation Pipeline Demonstration ===")

    # Generate test data with different scales and correlations
    np.random.seed(42)
    n_samples, n_features = 200, 4

    # Create correlated data
    base_data = np.random.randn(n_samples, n_features)
    base_data[:, 1] = base_data[:, 0] + 0.5 * np.random.randn(n_samples)  # Correlation
    base_data[:, 2] = -base_data[:, 0] + 0.3 * np.random.randn(n_samples)  # Anti-correlation

    # Different scales
    test_data = base_data.copy()
    test_data[:, 0] *= 100    # Large scale
    test_data[:, 1] += 50     # Shift
    test_data[:, 2] *= 0.01   # Small scale

    print(f"Original data shape: {test_data.shape}")
    print(f"Original data means: {np.mean(test_data, axis=0)}")
    print(f"Original data stds: {np.std(test_data, axis=0)}")

    # Initialize transformer
    transformer = DataTransformer()

    # Standardize data
    standardized_data = transformer.standardize(test_data)
    print(f"\nStandardized data means: {np.mean(standardized_data, axis=0)}")
    print(f"Standardized data stds: {np.std(standardized_data, axis=0)}")

    # Fit and apply PCA
    transformer.fit_pca(test_data, n_components=3)
    pca_data = transformer.transform_pca(test_data)

    pca_info = transformer.get_pca_info()
    print(f"\nPCA explained variance ratio: {pca_info['explained_variance_ratio']}")
    print(f"Cumulative variance explained: {pca_info['cumulative_variance_ratio']}")

    # Add polynomial features (small subset for demo)
    small_data = test_data[:10, :2]
    poly_data = transformer.add_polynomial_features(small_data, degree=2)
    print(f"\nPolynomial features: {small_data.shape} -> {poly_data.shape}")

    # Create interaction features
    interaction_data = transformer.create_interaction_matrix(small_data)
    print(f"Interaction features: {small_data.shape} -> {interaction_data.shape}")

    # Visualize transformations
    plt.figure(figsize=(15, 10))

    # Original data
    plt.subplot(2, 3, 1)
    plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.6)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data')

    # Standardized data
    plt.subplot(2, 3, 2)
    plt.scatter(standardized_data[:, 0], standardized_data[:, 1], alpha=0.6)
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.title('Standardized Data')

    # PCA data
    plt.subplot(2, 3, 3)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.6)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Transformed Data')

    # Explained variance
    plt.subplot(2, 3, 4)
    plt.bar(range(len(pca_info['explained_variance_ratio'])),
            pca_info['explained_variance_ratio'])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')

    # Cumulative explained variance
    plt.subplot(2, 3, 5)
    plt.plot(range(1, len(pca_info['cumulative_variance_ratio']) + 1),
             pca_info['cumulative_variance_ratio'], 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True)

    # Feature correlation heatmap
    plt.subplot(2, 3, 6)
    correlation_matrix = np.corrcoef(test_data.T)
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    plt.xlabel('Feature')
    plt.ylabel('Feature')

    plt.tight_layout()
    plt.show()
    print()


if __name__ == "__main__":
    demonstrate_matrix_calculator()
    demonstrate_image_processing()
    demonstrate_neural_network()
    demonstrate_data_transformer()

    print("=== Quiz Answers ===")
    print("1. Matrix multiplication [[1,2], [3,4]] @ [[5,6], [7,8]]:")
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = A @ B
    print(f"   Answer:\n{result}")

    print("\n2. Shape of (100, 5) @ (5, 3):")
    print(f"   Answer: (100, 3)")

    print("\n3. Data matrix shape for 1000 samples, 10 features:")
    print(f"   Answer: (1000, 10) - rows are samples, columns are features")

    # Additional verification
    print("\n=== Verification Tests ===")
    calc = MatrixCalculator()

    # Test matrix multiplication
    test_A = np.array([[1, 2], [3, 4]])
    test_B = np.array([[5, 6], [7, 8]])
    manual_result = np.array([[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]])
    calc_result = calc.multiply(test_A, test_B)
    print(f"Matrix multiplication verification: {np.array_equal(calc_result, manual_result)}")

    # Test inverse
    test_inv = calc.inverse(test_A)
    identity_check = test_A @ test_inv
    print(f"Matrix inverse verification: {np.allclose(identity_check, np.eye(2))}")