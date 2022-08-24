# Quant 

## Black-Scholes

-   Originally to valuate European call options
-   American equivalents: Bjerksund-Stendland model, binomial, trinomial
    models
-   Uses 5 Factors:
    1.  Volatility
    2.  Price of underlying asset
    3.  Strike price
    4.  Time to expiration
    5.  Risk free interest rate

-   **Black-Scholes Asumptions**
    -   Price follows a random walk approximately Geometric brownian
        motion with constant drift and volatility (i.e. log(variance) is
        constant)
    -   No dividends over life of option
    -   Movements are random, market is random
    -   No transaction costs
    -   RFR and volatility are constant (not a strong assumption for
        volatility, since that is influenced by supply/demand)
    -   Returns are log normal
    -   Option is European (can only be exercised at expiration)


## Greeks

1.  **Delta**

    First derivative with respect to price. Rate of change of
    equilibrium price (aka BS price) with respect to asset price.

2.  **Gamma**

    Second derivative with respect to price.

3.  **Theta**

    First derivative with respect to time-to-maturity. Rate of change of
    equilibrium price with respect to time-to-maturity.

4.  **Vega**

    Rate of change of equilibrium price with respect to asset
    volatility.

5.  **Rho**

    Rate of change of equilibrium price with respect to RF interest
    rate.

## Pay Off Diagrams

-   Plot of **Underlying Price vs. P&L**
-   3 Key Points:
    1.  Maximum Loss
    2.  Maximum Gain
    3.  Break-even Point

## Call Options

-   **Break-even**: K + P (where K is strike price and P is cost of
    option)
-   **5 reasons to buy a call option**
    1.  Bet on upside move with minimal cost (lot of a exposure for
        little cost)
    2.  Unlimited Upside
    3.  Limited Downside: Can only lose what you paid for the option
    4.  Increase in Volatility: Option is priced based on its
        volatility, so all we need is an increase in volatility to
        increase the value of our option
    5.  Hedge Short Position: Unlimited upside offsets risk of short as
        shorts have unlimited downside 

### Call-Spread

-   Max Value: difference in strike prices. $v_{max} = K_2 - K_1$
    -   Where $K_2=Sold$ and $K_1=bought$ strike prices
-   Max Loss: $\text{max_loss} = \text{max_value} -P_{cs}$
    -   Max value - Price of call-spread


## Put-Call Parity

-   Represents an arbitrage opportunity
-   $\text{call_price}+\text{present_value_discounted} = \text{put_price} + \text{spot_price}$
    -   (where present value is discounted from the value at RFR)

## Questions

-   What factors in production could cause a backtested strategy to
    perform different than expected?
    -   Slippage, transaction costs, systemic risk, outside events that
        cannot be modeled such as state of global
        economy/climate/legislation/etc