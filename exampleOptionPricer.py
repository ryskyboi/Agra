from math import exp
import matplotlib.pyplot as plt
import numpy as np
from optionPricing import  OptionPricer


vol = 0.7
eth_price = 2500
eth_collateral = 100
lev = 2
eth_supplied = eth_collateral * lev
usc_borrowed = (eth_supplied - eth_collateral) * eth_price

print(f"eth_supplied: {eth_supplied}")
print(f"usc_borrowed: {usc_borrowed}")
print(f"collareral_value: {eth_collateral * eth_price}")

rate_usdc = 0.05
rate_eth = 0.02
liquidation_threshold = 0.83
_time = 0.025  ## This is in years
loan_amount = 50_000 # 50k borrow against 100 eth loop
loan_time = 0.1
risk_free_rate = 0.05


pricer = OptionPricer()

def liquidation_price(supplied_amount: float, borrowed_price: float, borrowed_amount: float, rate_supplied: float, rate_borrowed: float, _time: float, liquidation_threshold: float) -> float:
    return ((borrowed_amount * borrowed_price) / (supplied_amount * liquidation_threshold)) * exp((rate_borrowed - rate_supplied) * _time)

def indifference_point(net_value: float, supplied_amount: float, borrowed_price: float, borrowed_amount: float, rate_supplied: float, rate_borrowed: float, _time: float)  -> float:
    return (net_value + borrowed_amount * borrowed_price * exp(rate_borrowed * _time)) / (supplied_amount * exp(rate_supplied * _time))

def rate_to_apr(rate: float, time: float) -> float:
    return (1+rate) ** (1/time) - 1

def example_loan_cost(eth_price: float, point_of_indifference: float, liquidation_point: float, loan_time: float, risk_free_rate: float, vol: float, loan_size: float, eth_supplied) -> float:
    down_and_out_put = pricer.down_and_out_put(eth_price, point_of_indifference, liquidation_point, loan_time, risk_free_rate, vol)
    american_digital_put = pricer.american_digital_put(eth_price, liquidation_point, loan_time, risk_free_rate, vol)
    if down_and_out_put > pricer.vanilla_put(eth_price, point_of_indifference, loan_time, risk_free_rate, vol):
        raise ValueError("Down and out put is worth more than vanilla put")
    return american_digital_put * loan_size + down_and_out_put * eth_supplied

liq_point_example = liquidation_price( eth_supplied, 1, usc_borrowed, rate_eth, rate_usdc, _time, liquidation_threshold)
indifference_point_example =  indifference_point( loan_amount, eth_supplied, 1, usc_borrowed, rate_eth, rate_usdc, _time)
option_cost = example_loan_cost(eth_price, indifference_point_example, liq_point_example, loan_time, risk_free_rate, vol, loan_amount, eth_supplied)
loan_cost = option_cost + loan_amount * (1 - exp(-rate_usdc * loan_time))
loan_cost_as_rate = loan_cost / loan_amount

print(f'Option cost: {option_cost}$')
print(f"Loan cost: {loan_cost}$")
print(f"Loan cost as percentage: {loan_cost_as_rate * 100}%")
print(f"Loan apr: {rate_to_apr(loan_cost_as_rate, loan_time) * 100}%")