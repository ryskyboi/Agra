from genealPricer import MonteCarloPricer
import numpy as np
from math import exp
import pandas as pd

monte = MonteCarloPricer()

eth_supplied = 300
usc_borrowed = 300_000
rate_usdc = 0.05
rate_eth = 0.02
liquidation_threshold = 0.83
_time = 0.025  ## This is in years
loan_amount = 150_000
vol = 0.7
eth_price = 2500
loan_time = 0.1
risk_free_rate = 0.05

def rate_to_apr(rate: float, time: float) -> float:
    return (1+rate) ** (1/time) - 1

'''
def is_liquidatable(supplied_price: float, supplied_amount: float, borrowed_price: float, borrowed_amount: float, rate_supplied: float, rate_borrowed: float, _time: float, liquidation_threshold: float) -> bool:
    return liquidation_threshold < (borrowed_price * borrowed_amount) / (supplied_price * supplied_amount) * exp((rate_borrowed - rate_supplied) * _time)

def liquidation_price(supplied_amount: float, borrowed_price: float, borrowed_amount: float, rate_supplied: float, rate_borrowed: float, _time: float, liquidation_threshold: float) -> float:
    return ((borrowed_amount * borrowed_price) / (supplied_amount * liquidation_threshold)) * exp((rate_borrowed - rate_supplied) * _time)

def account_value(supplied_price: float, supplied_amount: float, borrowed_price: float, borrowed_amount: float, rate_supplied: float, rate_borrowed: float, _time: float, liquidation_threshold: float) -> float:
    if is_liquidatable(supplied_price, supplied_amount, borrowed_price, borrowed_amount, rate_supplied, rate_borrowed, _time, liquidation_threshold):
        return 0
    return (supplied_price * supplied_amount) * exp(rate_supplied * _time) - (borrowed_price * borrowed_amount) * exp(rate_supplied * _time)



def payoff_aave(eth_prices: np.ndarray) -> np.ndarray:
    ## This will just be a 1d array for a list of prices
    results = np.full(len(eth_prices[0]), np.nan)
    for i, price in enumerate(eth_prices[0]):
        if price < liq_point_example:
            results[i:] = 0
            return results
        results[i] = min(loan_amount, account_value(price, eth_supplied, 1, usc_borrowed, rate_eth, rate_usdc, _time, liquidation_threshold))
    return results

liq_point_example = liquidation_price( eth_supplied, 1, usc_borrowed, rate_eth, rate_usdc, _time, liquidation_threshold)



mu, sigma, sim, walk = monte.evaluate(
    payoff_aave,
    [eth_price],
    [0],
    [vol],
    np.array([[1]]),
    1000,
    loan_time,
    risk_free_rate,
    100_000
)

print(f"loan cost : {loan_amount - mu * exp(-risk_free_rate * loan_time)}")
print(f"Rate : {(1 - (mu / loan_amount) * exp(-risk_free_rate * loan_time))}%")
print(f"APR : {rate_to_apr(1 - (mu / loan_amount) * exp(-risk_free_rate * loan_time), loan_time) * 100}%")
print(f"Standard Deviation : {sigma}")

rate_to_apr(1 - (mu / loan_amount) * exp(-risk_free_rate * loan_time), loan_time) * 100

'''

rsETH = pd.read_csv("data/RSETH_1Y_graph_coinmarketcap.csv", delimiter=";")
weth = pd.read_csv("data/WETH_1Y_graph_coinmarketcap.csv", delimiter=";")
corr = np.corrcoef(rsETH["close"], weth["close"])

rsETH_price = rsETH["close"].values[-1]
wETH_price = weth["close"].values[-1]

print(rsETH_price, wETH_price)


eth_vol = 0.65 ## normal

rseth_borrowed = 96
weth_supplied = 99.71 + 100


rseth_rate = 0.0368
weth_rate = 0.0176
liquidation_threshold = 0.93
loan_amount = 150_000

def euler_liquidation(rseth_price: float, rseth_borrowed: float, weth_price: float, weth_supplied: float, rseth_rate: float, rate_eth: float, liquidation_threshold: float, time: float) -> float:
    return (rseth_price * rseth_borrowed) / (weth_price * weth_supplied) * exp((rseth_rate - rate_eth) * time) > liquidation_threshold

def payoff_euler(prices: np.ndarray) -> np.ndarray:
    results = np.full(len(prices[0]), np.nan)
    loan_delta = loan_time / (len(prices[0]) - 1)
    for i in range(len(prices[0])):
        rs_price = prices[0][i]
        weth_price = prices[1][i]
        if euler_liquidation(rs_price, rseth_borrowed, weth_price, weth_supplied, rseth_rate, weth_rate, liquidation_threshold,  loan_delta * i):
            results[i:] = 0
            return results
        results[i] = min(loan_amount, weth_price * weth_supplied * exp(weth_rate * loan_delta * i) - rs_price * rseth_borrowed * exp(rseth_rate * loan_delta * i))
    return results

mu, sigma, sim, walk  = monte.evaluate(
    payoff_euler,
    [rsETH_price, wETH_price],
    [0.0229, 0], ## rates
    [0.8, 0.8],  ## Both have the same vol
    corr,
    1000,
    loan_time,
    risk_free_rate,
    100_000
)

print(f"loan cost : {loan_amount - mu * exp(-risk_free_rate * loan_time)}")
print(f"Rate : {(1 - (mu / loan_amount) * exp(-risk_free_rate * loan_time))}%")
print(f"APR : {rate_to_apr(1 - (mu / loan_amount) * exp(-risk_free_rate * loan_time), loan_time) * 100}%")
print(f"Standard Deviation : {sigma}")