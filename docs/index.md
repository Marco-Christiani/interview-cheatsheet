<!-- [Machine Learning](ml/ml.md) -->
<!-- [Quant](quant.md) -->
<!-- [Finance Fundamentals](finfun.md) -->
<!-- [Word Problems](wordprobs.md) -->
<!-- [Code Puzzles](codepuzzles.md) -->

# Math/Statistics 

## Probability vs. Likelihood

-   **Probability**: During the testing phase, given a learned model, we
    determine the probability of observing the outcome
-   **Likelihood**: During training, given some outcome we determine the
    likelihood of observing theta that maximizes the probability of that
    outcome (MLE).

## Bayes Theorem

-   Revise prediction using new evidence
-   **Naive Bayes Classifier**: Generative Classification model

## Linear Transformation

-   Rotation (produced by shearing) and scaling

\[
    f(\alpha x + \beta y)=\alpha f(x)+\beta f(y)
\]


## Affine Transformation

AKA a linear transformation plus translation 

\[
    f(\alpha x + (1-\alpha)y)=\alpha f(x)+(1-\alpha)f(y)
\]

\[
    f(x)=\vec{a}^Tx+\vec{b}
\]

**Properties Perserved:**

1.  Collinearity between points: points on same line (collinear points)
    remain on same line after transformation
2.  Parallelism: Parallel lines remain parallel
3.  Convexity of sets: Furthermore, extreme points of original set map
    to extreme points of transformed set
4.  Length Ratios of Parallel lines
5.  Barycenters of weighted collections of points. Aka center of mass. **?**

