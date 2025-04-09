import numpy as np
from typing import Callable
from tqdm import tqdm

class MonteCarloPricer:
    def multi_dimensional_random_walk(self, initial_prices: list[float], drifts: list[float], volatilitys: list[float],
                                    correlation_matrix: np.ndarray, num_points: int, dt: float, num_scenarios: int = 1000) -> np.ndarray:
        """
        Generate multiple correlated random walks for multiple assets.

        Parameters:
        initial_prices: Starting prices for each asset
        drifts: Drift rates for each asset
        volatilitys: Volatility for each asset
        correlation_matrix: Matrix of correlations between assets (n_assets x n_assets)
        num_points: Number of time points to simulate
        dt: Time step
        num_scenarios: Number of scenarios to simulate

        Returns a 3D array with shape (num_assets, num_scenarios, num_points)
        """
        num_assets = len(initial_prices)
        if len(initial_prices) != len(drifts) or len(initial_prices) != len(volatilitys):
            raise ValueError("Price, drift, and volatility lists must be the same length")
        if correlation_matrix.shape != (num_assets, num_assets):
            raise ValueError(f"Correlation matrix must be {num_assets}x{num_assets}")

        # Create a 3D array with shape (num_assets, num_scenarios, num_points)
        result = np.zeros((num_assets, num_scenarios, num_points))

        # Initialize the first point in time across all scenarios with initial prices
        result[:, :, 0] = np.tile(np.array(initial_prices).reshape(-1, 1), (1, num_scenarios))

        # Convert lists to numpy arrays for vectorized operations
        drifts_array = np.array(drifts)
        variances_array = np.array(volatilitys) ** 2

        # Compute Cholesky decomposition of the correlation matrix
        try:
            cholesky_matrix = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If correlation matrix is not positive definite, add a small value to diagonal
            adjusted_matrix = correlation_matrix + np.eye(num_assets) * 1e-5
            cholesky_matrix = np.linalg.cholesky(adjusted_matrix)

        # For each time step
        for i in tqdm(range(1, num_points)):
            # Get previous prices
            prev_prices = result[:, :, i-1]

            # Generate uncorrelated standard normal random values
            uncorrelated_random = np.random.standard_normal((num_assets, num_scenarios))

            # Transform to correlated random values using Cholesky decomposition
            # Apply cholesky_matrix @ uncorrelated_random for each scenario
            correlated_random = np.zeros((num_assets, num_scenarios))
            for j in range(num_scenarios):
                correlated_random[:, j] = cholesky_matrix @ uncorrelated_random[:, j]

            # Apply GBM formula with correlated random values
            result[:, :, i] = prev_prices * np.exp(
                (drifts_array.reshape(-1, 1) - 0.5 * variances_array.reshape(-1, 1)) * dt +
                np.sqrt(variances_array.reshape(-1, 1) * dt) * correlated_random
            )

        return result

    def evaluate(self, payoff_function: Callable[[np.ndarray], np.ndarray], initial_prices: list[float], drifts: list[float],
                    volatilitys: list[float], corr: np.ndarray, num_points: int, time: float, risk_free_rate: float, scenarios: int = 1000):
        if np.all(np.isnan(drifts)):
            drifts = list(np.full(len(drifts), risk_free_rate))
        dt = time/(num_points - 1)
        walks = self.multi_dimensional_random_walk(initial_prices, drifts, volatilitys, corr, num_points, dt, scenarios)
        sim = np.empty((scenarios, num_points))
        for j in tqdm(range(scenarios)):
            ## Rows are assets, times are columns
            sim[j,:] = payoff_function(walks[:,j,:])
        return np.mean(sim[:,-1]), np.std(sim[:,-1]), sim, walks
