import numpy as np
from typing import Callable
from tqdm import tqdm

class MonteCarloPricer:
    def multi_dimensional_random_walk(self, initial_prices: list[float], drifts: list[float], variances: list[float],
                                       num_points: int, dt: float, num_scenarios: int = 1000) -> np.ndarray:
        """
        Generate multiple random walks for multiple assets.

        Returns a 3D array with shape (num_assets, num_scenarios, num_points)
        """
        if len(initial_prices) != len(drifts) or len(initial_prices) != len(variances):
            raise ValueError("lists must be the same length")

        # Create a 3D array with shape (num_assets, num_scenarios, num_points)
        result = np.zeros((len(initial_prices), num_scenarios, num_points))

        # Initialize the first point in time across all scenarios with initial prices
        result[:, :, 0] = np.tile(np.array(initial_prices).reshape(-1, 1), (1, num_scenarios))

        # Convert lists to numpy arrays for vectorized operations
        drifts_array = np.array(drifts)
        variances_array = np.array(variances)

        # Generate all random normal values at once (num_assets, num_scenarios, num_points-1)
        random_values = np.random.standard_normal((len(initial_prices), num_scenarios, num_points-1))

        for i in range(1, num_points):
            # Get previous prices
            prev_prices = result[:, :, i-1]

            # Calculate drift component: μΔtS_t
            drift_component = drifts_array.reshape(-1, 1) * dt * prev_prices

            # Calculate volatility component: σ√ΔtS_tY_i
            volatility_component = (np.sqrt(variances_array * dt).reshape(-1, 1) *
                                   prev_prices * random_values[:, :, i-1])

            # New price = old price + drift + random shock
            result[:, :, i] = prev_prices + drift_component + volatility_component

        return result
    
    def evaluate(self, payoff_function: Callable[[np.ndarray], np.ndarray], initial_prices: list[float], drifts: list[float],
                    variances: list[float], num_points: int, time: float, scenarios: int = 1000):
        ##TODO: Actually this payoff function needs to also take the state of all values in the array at that time
        dt = time/(num_points - 1)
        walks = self.multi_dimensional_random_walk(initial_prices, drifts, variances, num_points, dt, scenarios)
        sample_means, sample_variances = np.zeros(len(initial_prices)), np.zeros(len(initial_prices))
        sim = np.zeros((len(initial_prices), scenarios, num_points))
        for i in range(len(initial_prices)):
            print(f"Simulating scenarios for asset: {i}")
            for j in tqdm(range(scenarios)):
                sim[i,j,:] = payoff_function(walks[i,j,:])
            sample_means[i] = np.mean(sim[i, :, -1])
            sample_variances[i] = np.var(sim[i, :, -1])
        return sample_means, sample_variances, sim, walks

