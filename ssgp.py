import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve

class SSGP:
    def __init__(self, n_features=100, length_scale=1.0, period=1.0, sigma_f=1.0, sigma_n=0.1, kernel="RBF"):
        """
        Initialize the SSGP model.

        Parameters:
        - n_features: Number of Fourier features to use.
        - length_scale: Length scale of the kernel.
        - sigma_f: Signal variance.
        - sigma_n: Noise variance.
        - kenerl: kernel option(RBF, ExpSin, ExpSinSquared)
        """
        self.n_features = n_features
        self.length_scale = length_scale
        self.period = period
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.kernel = kernel
        self.weights = None
        self.random_features = None
        self.omega = None
        self.bias = None
        self.A_inv = None  # Store (K + σ²I)^-1 for predictive variance

    def _generate_random_features_RBF(self, X):
        """
        Generate random Fourier features.

        Parameters:
        - X: Input data (n_samples, n_features).

        Returns:
        - Random features matrix (n_samples, n_features).
        """
        n_samples, n_dim = X.shape
        if self.omega is None:
            # omega is drawn from multivariate normal distribution to approximate RBF kernel
            self.omega = np.random.normal(0, 1 / self.length_scale, (n_dim, self.n_features))
            self.bias = np.random.uniform(0, 2 * np.pi, self.n_features)
        phi = np.sqrt(2 * self.sigma_f / self.n_features) * np.cos(X @ self.omega + self.bias)
        return phi

    def _generate_random_features_ExpSin(self, X):
        n, d = X.shape
        # Sample frequencies from an appropriate distribution
        # Here, we use a combination of Gaussian (for decay) and uniform (for periodicity)
        omega = np.random.normal(0, 1 / self.length_scale, (self.n_features, d))  # Gaussian for decay
        omega += np.random.uniform(-np.pi / self.period, np.pi / self.period, (self.n_features, d))  # Uniform for periodicity
        # Compute the feature map
        phi = np.sqrt(2 / self.n_features) * np.cos(X @ omega.T)
        return phi

    def _generate_random_features_ExpSinSquared(self, X):
        n, d = X.shape
        # Sample frequencies from the spectral density of the Exponential Sine Squared Kernel
        # The spectral density is a mixture of Gaussians centered at multiples of the fundamental frequency
        omega = np.random.normal(0, 1 / self.length_scale, (self.n_features, d))  # Gaussian for decay
        omega += np.random.choice([-1, 1], (self.n_features, d)) * (2 * np.pi / self.period)  # Periodic component
        # Compute the feature map
        phi = np.sqrt(2 / self.n_features) * np.cos(X @ omega.T)
        return phi

    def fit(self, X, y):
        """
        Fit the SSGP model to the data.

        Parameters:
        - X: Input data (n_samples, n_features).
        - y: Target values (n_samples,).
        """

        if self.kernel == "RBF":
            self.random_features = self._generate_random_features_RBF(X)
        if self.kernel == "ExpSin":
            self.random_features = self._generate_random_features_ExpSin(X)
        if self.kernel == "ExpSinSquared":
            self.random_features = self._generate_random_features_ExpSinSquared(X)
            
        A = self.random_features.T @ self.random_features + (self.sigma_n ** 2) * np.eye(self.n_features)
        self.A_inv = cho_solve(cho_factor(A), np.eye(self.n_features))  # Store (K + σ²I)^-1
        self.weights = self.A_inv @ self.random_features.T @ y


    def predict(self, X, return_std=True):
        """
        Predict using the SSGP model.

        Parameters:
        - X: Input data (n_samples, n_features).
        - return_std: If True, return the standard deviation of the predictions.

        Returns:
        - Predicted values (n_samples,).
        - Standard deviation of the predictions (n_samples,) (if return_std=True).
        """
        if self.kernel == "RBF":
            phi = self._generate_random_features_RBF(X)
        if self.kernel == "ExpSin":
            phi = self._generate_random_features_ExpSin(X)
        if self.kernel == "ExpSinSquared":
            phi = self._generate_random_features_ExpSinSquared(X)
        
        y_pred = phi @ self.weights

        if return_std:
            # Compute predictive variance
            var_f = self.sigma_f ** 2 - np.sum(phi @ self.A_inv * phi, axis=1)
            var_f = np.maximum(var_f, 0)  # Ensure non-negative variance
            std_f = np.sqrt(var_f + self.sigma_n ** 2)  # Add noise variance
            return y_pred, std_f
        else:
            return y_pred

    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute the kernel matrix between two sets of input points.

        Parameters:
        - X1: Input data (n_samples1, n_features).
        - X2: Input data (n_samples2, n_features). If None, X2 = X1.

        Returns:
        - Kernel matrix (n_samples1, n_samples2).
        """
        if X2 is None:
            X2 = X1
            
        if self.kernel == "RBF":
            phi1 = self._generate_random_features_RBF(x1.reshape(1, -1))
            phi2 = self._generate_random_features_RBF(x2.reshape(1, -1))        
        if self.kernel == "ExpSin":
            phi1 = self._generate_random_features_ExpSin(x1.reshape(1, -1))
            phi2 = self._generate_random_features_ExpSin(x2.reshape(1, -1))
        if self.kernel == "ExpSinSquared":
            phi1 = self._generate_random_features_ExpSinSquared(x1.reshape(1, -1))
            phi2 = self._generate_random_features_ExpSinSquared(x2.reshape(1, -1))
            
        return phi1 @ phi2.T

    def true_kernel(self, x1, x2):
        """
        Compute the true squared exponential kernel.

        Parameters:
        - x1, x2: Input points.

        Returns:
        - Kernel value.
        """
        if self.kernel == "RBF":
            return self.sigma_f ** 2 * np.exp(-0.5 * np.sum((x1 - x2) ** 2) / self.length_scale ** 2)
        if self.kernel == "ExpSin":
            r = np.abs(x1 - x2)
            return np.exp(-np.abs(np.sin(np.pi * r / self.period)) / self.length_scale)
        if self.kernel == "ExpSinSquared":
            r = np.abs(x1 - x2)
            return np.exp(-2 * np.sin(np.pi * r / self.period)**2 / self.length_scale**2)
            

    def approximate_kernel(self, x1, x2):
        """
        Compute the approximate kernel using random Fourier features.

        Parameters:
        - x1, x2: Input points.

        Returns:
        - Approximate kernel value.
        """
        if self.kernel == "RBF":
            phi1 = self._generate_random_features_RBF(x1.reshape(1, -1))
            phi2 = self._generate_random_features_RBF(x2.reshape(1, -1))        
        if self.kernel == "ExpSin":
            phi1 = self._generate_random_features_ExpSin(x1.reshape(1, -1))
            phi2 = self._generate_random_features_ExpSin(x2.reshape(1, -1))
        if self.kernel == "ExpSinSquared":
            phi1 = self._generate_random_features_ExpSinSquared(x1.reshape(1, -1))
            phi2 = self._generate_random_features_ExpSinSquared(x2.reshape(1, -1))
            
        return (phi1 @ phi2.T).item()  # Convert to scalar

    def compute_mean_squared_error(y_true, y_pred, title="Mean Squared Error"):
        """
        Compute and plot the Mean Squared Error (MSE) between true and predicted values.
    
        Parameters:
        - y_true: True target values (n_samples,).
        - y_pred: Predicted values (n_samples,).
        """
        # Compute MSE
        mse = np.mean((y_true - y_pred) ** 2)
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        return mse
        
        # # Plot true vs predicted values
        # plt.figure(figsize=(8, 6))
        # plt.scatter(y_true, y_pred, color="blue", label="True vs Predicted")
        # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", linestyle="--", label="Ideal Fit")
        # plt.xlabel("True Values")
        # plt.ylabel("Predicted Values")
        # plt.title(f"{title}\nMSE: {mse:.4f}")
        # plt.legend()
        # plt.grid(True)
        # plt.show()    
        



