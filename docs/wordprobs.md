# Word Problems 

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

