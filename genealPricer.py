import numpy as np
from typing import Callable
from tqdm import tqdm

class MonteCarloPricer:
    def multi_dimensional_random_walk(self, initial_prices: list[float], drifts: list[float], volatilitys: list[float],
                                       num_points: int, dt: float, num_scenarios: int = 1000) -> np.ndarray:
        """
        Generate multiple random walks for multiple assets.

        Returns a 3D array with shape (num_assets, num_scenarios, num_points)
        """
        if len(initial_prices) != len(drifts) or len(initial_prices) != len(volatilitys):
            raise ValueError("lists must be the same length")

        # Create a 3D array with shape (num_assets, num_scenarios, num_points)
        result = np.zeros((len(initial_prices), num_scenarios, num_points))

        # Initialize the first point in time across all scenarios with initial prices
        result[:, :, 0] = np.tile(np.array(initial_prices).reshape(-1, 1), (1, num_scenarios))

        # Convert lists to numpy arrays for vectorized operations
        drifts_array = np.array(drifts)
        variances_array = np.array(volatilitys) ** 2

        # Generate all random normal values at once (num_assets, num_scenarios, num_points-1)
        random_values = np.random.standard_normal((len(initial_prices), num_scenarios, num_points-1))

        for i in range(1, num_points):
            # Get previous prices
            prev_prices = result[:, :, i-1]

            # This uses GMB instead
            result[:, :, i] = prev_prices * np.exp((drifts_array.reshape(-1, 1) - 0.5 * variances_array.reshape(-1, 1)) * dt +
                                     np.sqrt(variances_array.reshape(-1, 1) * dt) * random_values[:, :, i-1])
        return result

    def evaluate(self, payoff_function: Callable[[np.ndarray], np.ndarray], initial_prices: list[float], drifts: list[float],
                    volatilitys: list[float], num_points: int, time: float, risk_free_rate: float, scenarios: int = 1000):
        if np.all(np.isnan(drifts)):
            drifts = list(np.full(len(drifts), risk_free_rate))
        dt = time/(num_points - 1)
        walks = self.multi_dimensional_random_walk(initial_prices, drifts, volatilitys, num_points, dt, scenarios)
        sim = np.empty((scenarios, num_points))
        for j in tqdm(range(scenarios)):
            ## Rows are assets, times are columns
            sim[j,:] = payoff_function(walks[:,j,:])
        return np.mean(sim[:,-1]), np.std(sim[:,-1]), sim, walks
