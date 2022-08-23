Math/Statistics 
===============

Probability vs. Likelihood
--------------------------

-   **Probability**: During the testing phase, given a learned model, we
    determine the probability of observing the outcome
-   **Likelihood**: During training, given some outcome we determine the
    likelihood of observing theta that maximizes the probability of that
    outcome (MLE).

Bayes Theorem
-------------

-   Revise prediction using new evidence
-   **Naive Bayes Classifier**: Generative Classification model

Linear Transformation
---------------------

$f(\alpha x + \beta y)=\alpha f(x)+\beta f(y)$

-   Linear transformations: rotation (produced by shearing), scaling

Affine Transformation
---------------------

$$f(\alpha x + (1-\alpha)y)=\alpha f(x)+(1-\alpha)f(y)$$ AKA a linear
transformation plus translation $$f(x)=\vec{a}^Tx+\vec{b}$$ **Properties
Perserved:**

1.  Collinearity between points: points on same line (collinear points)
    remain on same line after transformation
2.  Parallelism: Parallel lines remain parallel
3.  Convexity of sets: Furthermore, extreme points of original set map
    to extreme points of transformed set
4.  Length Ratios of Parallel lines
5.  Barycenters of weighted collections of points. Aka center of mass.
    **?**

Machine Learning 
================

Discriminative Model
--------------------

-   learn decision boundaries between classes
-   SVM, Logistic Regression, Decision Trees
-   Not great w outliers
-   **Maximizes conditional likelihood**, given model parameters
    -   L(theta) = max P(y \| x; theta)

Generative Model
----------------

-   distribution of classes themselves
-   Naïve Bayes, Discriminant Analysis (LDA, GDA)
-   Better with outliers
-   **Maximizes joint likelihood:** the joint probability given model
    parameters
    -   L(theta)=max P(x, y; theta)

Cross Validation
----------------

-   Typically use k-fold validation: i.e. leave one out cross validation
-   **Roll forward Cross Validation**: used with Time Series Data

Tree Pruning
------------

-   Mitigate Overfitting

### Cost Effective Pruning:

-   Remove a subtree (replacing with a leaf node)
-   if resulting tree does not have a significant decrease in
    performance (delta formula) then keep the new pruned tree and
    repeat.

Dealing with Datasets
---------------------

### **Imbalanced Datasets**

-   **Random Under-sampling**: Lots of data in smaller class
-   **Random Over-sampling**: Not lots of data in smaller class

### Missing Data:

-   Imputation (i.e. 0), add a new category for categorical (I.e.
    "other"), interpolation

### Outliers

-   \*Analyze without and without outliers
-   Trimming: Remove outliers
-   Winsorizing: Ceil/Floor to a max/min non-outlier value

SMOTE
-----

-   Synthetic Monetary Oversampling\*
-   Synthesize new data with minor noise added to existing sample rather
    than exact copies

ROC {#roc id="6304ff68-bb00-4200-aa44-975f5ef5aba9" collapsed="true"}
---

-   Receiver Operator Characteristic
-   Graphs Sensitivity vs Specificity (OR Precision)
-   i.e. True Positive vs True Negative Rates

![](../assets/unnamed_1661273383447_0.png)

### Accuracy

True predictions/Number points

### Precision

precision=TP/(TP+FP)

-   How many of our positive predictions were right?
-   Positive Prediction Accuracy for the label
-   Proportion of positive results that were correctly classified
-   $\text{precision}=\text{true\_pos}/(\text{true\_pos} + \text{true\_neg})$
-   Good if we have an imbalance such as way more negatives than
    positives (not in eq)

### Sensitivity/Recall

```{=latex}
\begin{align}
    \text{sensitivity}&=TP/(TP+FN)\\
    &=\text{true_pos}/(\text{true_pos} + \text{false_neg})
\end{align}
```
-   Returns the True Positive Rate of the label.

### Specificity: TN/(TN+FP)

$$\text{specificity}=\text{false\_pos}/(\text{false\_pos} + \text{true\_neg})$$

AUC
---

-   Area Under the Curve
-   Used to compare ROC curves
-   More AUC=better

Neural Networks
---------------

### RNN

-   Handle sequential data (unlike feedforward nn).
-   Sentiment analysis, text mining, image captioning, time series
    problems

### CNNs

-   Image matrix, filter matrix
-   Slide filter matrix over the image acompute the dot product to get
    convolved feature matrix.
-   **CNN better than Dense NN for Images:** Because less params (no
    overfit), more interpretable (can look at weights), CNNs can learn
    simple-to-complex patterns (learn complex patterns by learning
    several simple patterns)

### GANs

-   Use a Generator and Discriminator (to build an accurate
    Discriminator model)

### Activation Functions

1.  Softmax

    -   Scales input to (0,1). Output layers

2.  ReLU

    -   Clips input at 0, only non-negative outputs.
    -   Produces "rectified feature map." Hidden layers

3.  Swish

    -   Variant of ReLU developed at google, better at some DL tasks

### Pooling

-   Pooling is a down-sampling operation that reduces the dimensionality
    of the feature map

### Computation Graph

-   Nodes are operations, Edges are tensors/data

### Batch Gradient vs Stochastic Gradient Descent

### Autoencoder {#autoencoder collapsed="true"}

-   3 Layer model that tries to reconstruct its input using a hidden
    layer of fewer dimensions to create a latent space representation.
-   In its most basic form, uses dimensionality reduction to perform
    filtering (i.e. noise).

1.  Regularized Autoencoders:

    Classification (include Sparse, Denoising, Contractive)

2.  Variational Autoencoders:

    Generative models

3.  Uses

    -   Extract features and hidden patterns
    -   Dimensionality reduction
    -   Denoise im ages
    -   Convert black and w hite images into colored images.

### Transfer Learning

-   Models: VGG-16, BERT, GPT-3, Inception, Xception

### Vanishing Gradients

-   Use ReLU instead of tanh (try different activation function)
-   Try Xavier initialization (takes into account number of inputs and
    outputs).

### ANN Hyperparameters

1.  **Batch size:** size of input data
2.  **Epochs:** number of times training data is visible to the neural
    network to train.
3.  **Momentum:** Dampen/attenuate oscillations in gradient descent. If
    the weights matrix is ill conditioned, this helps convergence speed
    up a bit.
4.  **Learning rate:** Represents the time required for the network to
    update the parameters and learn.

Ensemble learning
-----------------

-   Boosting, Bagging, Random Forest
-   Aggregation mitigates overfitting of a class

### Bagging

-   Train several models and vote to produce output.

### Boosting

-   Use a model to improve performance where another model is weakest.
    (i.e. model the error)

Logistic Regression
-------------------

-   Regression for classification.
-   Linear model produces logits, softmax(logits) produces prediction

Fourier Transform
-----------------

-   Decompose functions into its constituent parts.

Quant 
=====

-   What factors in production could cause a backtested strategy to
    perform different than expected?
    -   Slippage, transaction costs, systemic risk, outside events that
        cannot be modeled such as state of global
        economy/climate/legislation/etc

Black-Scholes
-------------

-   Originally to valuate European call options
-   American equivalents: Bjerksund-Stendland model, binomial, trinomial
    models
-   Uses 5 Factors:
    1.  Volatility
    2.  Price of underlying asset
    3.  Strike price
    4.  Time to expiration
    5.  Risk free interest rate

```{=html}
<!-- -->
```
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

### Options

### Option Greeks

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

### Call Options

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
        shorts have unlimited downside {{embed
        ((6304378a-0e5d-42bc-bfff-7a551ffc9d47))}}

### Call-Spread

-   Max Value: difference in strike prices. $v_{max} = K_2 - K_1$
    -   Where $K_2=Sold$ and $K_1=bought$ strike prices
-   Max Loss: $Loss_{max} = v_{max}-P_{cs}$
    -   Max value - Price of call-spread

### Pay Off Diagrams

-   Plot of **Underlying Price vs. P&L**
-   3 Key Points:
    1.  Maximum Loss
    2.  Maximum Gain
    3.  Break-even Point

### Put-Call Parity

-   Represents an arbitrage opportunity
-   $\text{call\_price}+\text{present\_value\_discounted} = \text{put\_price} + \text{spot\_price}$
    -   (where present value is discounted from the value at RFR)

Finance Fundamentals 
====================

**Financial Statements**
------------------------

### **Balance sheet**

Shows a company\'s...

1.  Assets
2.  Liabilities
3.  Shareholders\' equity (what it owns, owes, is worth)

-   Highlights: liquidity, capital assets, credit metrics, liquidity
    ratios, leverage, ROA (return on assets), ROW (return on equity)

### **Income Statement**

Shows a company\'s...

1.  Revenue
2.  Expenses
3.  Net Income

-   Highlights: Growth rates, margins, profitability

### **Cash Flow Statement**

Shows a company\'s cash inflows/outlflows from...

1.  Operation
2.  Investments
3.  Financing

-   Highlights: short/long term cash flow profile, needs to raise money
    or return capital to shareholders

### **What is the best financial statement to measure a company\'s health?**

-   Cash is king. *Cash Flow Statement* shows how much cash company is
    actually generating
-   Arguments for other statements:
-   Balance Sheet: assets are true driver of cash flow
-   Income Statement: Earning power and profitability on an accural
    basis

### WACC

-   Weighted Average Cost of Capital
-   Blended cost of capital across all sources (common/preferred shares,
    debt)
-   (% debt vs total capital) x (1-effective tax rate)+(% equity vs
    capital) x (required return on equity) **check this**

**What is cheaper: debt or equity?**
------------------------------------

-   Debt: backed by collateral and paid off before equity
-   Debt is more liquid **?**

Finance Formulas 
================

-   *Revenue* = Volume x Price
-   *Cost* = Fixed Cost + Variable Cost
-   *Profit* = Revenue - Cost
-   *Profitability or Profit Margin* = Profit/Revenue
-   *ROI* = Annual Profit / Principal Investment
-   *Breakeven or Payback Period* = Principal / Annual Profit
-   *ROE* = Profits / Shareholder Equity
-   *ROA* = Profits / Total Assets

Word Problems 
=============

-   **Given a random number generator which provides a random real value
    between 0 to 1, how can you estimate the value of pi?**
    -   Monte Carlo integration (unit circle inside a square, ratio of
        points in circle versus points outside)
-   **Find the minimum number of socks I need to take out from a box of
    red and black socks to ensure that I have k pairs of socks.**
    -   Use **Pigeon Hole Principle**
        1.  Pick up N socks (one of each color)
        2.  Next sock forms a pair
    -   Answer: **2k+N-1**
    -   Note: When coding ensure to check if K\>total~pairs~ in list
        (pairs+=arr\[i\]/2)
-   **Can you minimize piecewise linear function without adding
    auxiliary variables?**
    -   [See this
        lecture](https://www.seas.ucla.edu/~vandenbe/ee236a/lectures/pwl.pdf)
    -   Firstly: is the function convex
    -   Convex piecewise-linear (piecewise-affine is a more accurate
        term) can be expressed as:
        $$f(x)=\max _{i=1, \ldots, m}\left(a_{i}^{T} x+b_{i}\right)$$
    -   Problem becomes: $\min f(x)$
    -   Therefore minimize each:
        -   $\min t$ subject to $a_{i}^{T} x+b_{i} \le t$ for $i=1,..,m$
        -   Basically: **no**

Code Puzzles 
============

Reverse a Linked List 
---------------------

``` {.python}
Curr=head
while curr:
    Next = curr.next
    Curr.next = prev
    Prev = curr
    Curr = next
return prev (new head)
```

Longest Palindrome 
------------------

``` {.python}
def longest_palindrome(s: str):
    if not s:
       return “”
    longest = “”
    for i in range(len(s)):
           # odd case, like “aba”
         tmp = helper(s, i, i)
         if len(tmp) > len(longest):
              # update result
              longest = tmp

         # even case, like “abba”
         tmp = helper(s, i, i+1)
         if len(tmp) > len(longest):
              longest = tmp
   return longest

def helper(s: str, l: int, r: int):
    while l >= 0 and r < len(s) and s[l] == s[r]:
       l -= 1 #decrement the left
       r += 1 #increment the right
    return s[l+1:r]

```

BFS (not recursive) 
-------------------

``` {.python}
# Visit adjacent unvisited vertex. 
# - Mark it as visited. Display it. Insert it in a queue.
# If no adjacent vertex, pop vertex off queue
# Repeat Rule 1 and Rule 2 until the queue is empty. 

def bfs(graph, current_node):
    visited = []
    queue = [current_node]

    while queue:
        s = queue.pop(0)
        print(s)
        for neighbor in graph[s]:
            if neighbor not in visited:
                visited.append(neighbor)
                queue.append(neighbor)
bfs(graph, 'A')
```

DFS (not recursive) 
-------------------

``` {.python}
# add unvisited nodes to stack
def dfs(graph, start_vertex):
    visited = set()
    traversal = []
    stack = [start_vertex]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            traversal.append(vertex)
            stack.extend(reversed(graph[vertex])) # add in same order as visited
    return traversal

```

Check if binary trees are equal: 
--------------------------------

``` {.python}
def are_identical(root1, root2):
  if root1 == None and root2 == None:
    return True

  if root1 != None and root2 != None:
    return (root1.data == root2.data and
              are_identical(root1.left, root2.left) and
              are_identical(root1.right, root2.right))

  return False
```

Etc 
---

``` {.python}
```
