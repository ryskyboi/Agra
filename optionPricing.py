import numpy as np
from scipy.stats import norm


class OptionPricer:
    def _d1(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Calculate d1 parameter for Black-Scholes
        formula.
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility
        """
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Calculate d2 parameter for Black-Scholes
        formula.
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility
        """
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    def vanilla_put(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Calculate vanilla put option price using Black-Scholes formula.

        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility
        """
        if T <= 0:
            return max(K - S, 0)

        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def digital_put(self, S: float, K: float, T: float, r: float, sigma: float):
        """
        Calculate digital (cash-or-nothing) put option price.

        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility
        """
        if T <= 0:
            return 1.0 if S < K else 0.0

        d2 = self._d2(S, K, T, r, sigma)

        return np.exp(-r * T) * norm.cdf(-d2)

    def down_and_out_put(self, S: float, K: float, B: float, T: float, r: float, sigma: float):
        """
        Calculate down-and-out put option price.

        Parameters:
        S: Current stock price
        K: Strike price (must be > B)
        B: Barrier (must be < S)
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility
        """
        if B >= K:
            raise ValueError("Barrier B must be less than strike K")
        if S <= B:
            return 0  # Option is knocked out if S is below barrier

        # Calculate α parameter
        alpha = (r - 0.5 * sigma**2) / (sigma**2)

        # Calculate vanilla and digital put prices
        p_vanilla_k = self.vanilla_put(S, K, T, r, sigma)
        p_vanilla_b = self.vanilla_put(S, B, T, r, sigma)
        p_digital_b = self.digital_put(S, B, T, r, sigma)

        # Calculate terms with reflection principle
        reflection_factor = (S / B)**(2 * alpha)
        p_vanilla_k_reflect = self.vanilla_put((B**2) / S, K, T, r, sigma)
        p_vanilla_b_reflect = self.vanilla_put((B**2) / S, B, T, r, sigma)
        p_digital_b_reflect = self.digital_put((B**2) / S, B, T, r, sigma)

        # Final formula
        return (p_vanilla_k - p_vanilla_b - (K - B) * p_digital_b
                - reflection_factor * (p_vanilla_k_reflect - p_vanilla_b_reflect + (K - B) * p_digital_b_reflect))