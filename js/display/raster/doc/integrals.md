# Integrals over Polygons

## Green's Theorem and Polygons

Using [Green's Theorem](https://en.wikipedia.org/wiki/Green%27s_theorem), we can convert a double integral over a region into a line integral over the (closed, oriented counterclockwise) boundary of the region:

For curves parameterized on $`t`$:
$`
\oint\left(L\,\frac{dx}{dt}+M\,\frac{dy}{dt}\right)dt=\iint_P \left( \frac{\partial M}{\partial x}-\frac{\partial L}{\partial y} \right)\,dxdy
`$

For polygons, this means that if we can evaluate a line integral over each edge (point $`(x_i,y_i)`$ to point $`(x_{i+1},y_{i+1})`$), we can sum up each edge's contribution to get the double integral over the polygon.

There are two notable things:

1. If we reverse an edge (swap its endpoints), it will swap the sign of the contribution to the integral.
2. We are evaluating this on closed polygons, so any terms that only depend on one endpoint will cancel out (e.g. $`x_i^2y_i`$ and $`-x_{i+1}^2y_{i+1}`$ in summations will cancel out, and those terms will always be the additive inverse of each other$)

We can pick $`L`$ and $`M`$ below:

$`
L=(n-1)\int f\,dy
`$
$`
M=n\int f\,dx
`$
for any antiderivatives and real $`n`$, since the double integral will then be integrating our function $`f`$.

It turns out, evaluating Green's Theorem over line segments for polynomial terms for any linear blend (any $`n`$) of $`L`$ and $`M`$ will differ only in the "canceled out" edges, so they are all equivalent.

## Integrating Arbitrary Polynomials over Polygons

If we zero out all of the canceled terms, it turns out that we can evaluate the integral of any polynomial term $`x^my^n`$ over a polygon $`P`$ by summing up the contributions of each edge:

$`\iint_Px^my^n\,dxdy=\frac{m!n!}{(m+n+2)!}\sum_{i}\left[ (x_iy_{i+1}-x_{i+1}y_i) \sum_{p=0}^m\sum_{q=0}^n \binom{p+q}{q}\binom{m+n-p-q}{n-q}x_i^{m-p}x_{i+1}^py_i^{n-q}y_{i+1}^q \right]`$

The contributions of each term can be summed up individually to integrate arbitrary polynomials.

e.g. for $`x^4y^2`$ in matrix form:

$`
\iint_Px^4y^2\,dxdy=
\frac{1}{840}
(x_iy_{i+1}-x_{i+1}y_i)
\begin{bmatrix}
x_i^4 & x_i^3x_{i+1} & x_i^2x_{i+1}^2 & x_ix_{i+1}^3 & x_{i+1}^4
\end{bmatrix}
\begin{bmatrix}
15 & 5 & 1\\
10 & 8 & 3\\
6 & 9 & 6\\
3 & 8 & 10\\
1 & 5 & 15
\end{bmatrix}
\begin{bmatrix}
y_i^2\\
y_iy_{i+1}\\
y_{i+1}
\end{bmatrix}
`$

## Area of Polygons

For $`x^0y^0=1`$, we'll have the [Shoelace formula](https://en.wikipedia.org/wiki/Shoelace_formula) for finding the area of a polygon:

$`
area_P=\iint_P1\,dxdy=
\frac{1}{2}
\sum_{i}
(x_i+x_{i+1})(y_{i+1}-y_i)
`$

## Centroids of Polygons

For $`x`$ and $`y`$, we have:
$`
\iint_Px\,dxdy=
\frac{1}{6}
\sum_{i}
(x_iy_{i+1}-x_{i+1}y_i)(x_i+x_{i+1})
`$
$`
\iint_Py\,dxdy=
\frac{1}{6}
\sum_{i}
(x_iy_{i+1}-x_{i+1}y_i)(y_i+y_{i+1})
`$

where we can divide the integrals of $`x`$ and $`y`$ by the area to get the centroid of the polygon:
$`
centroid_x=
\frac{1}{3}
\frac{\sum_{i}(x_iy_{i+1}-x_{i+1}y_i)(x_i+x_{i+1})}{\sum_{i}(x_i+x_{i+1})(y_{i+1}-y_i)}
`$
$`
centroid_y=
\frac{1}{3}
\frac{\sum_{i}(x_iy_{i+1}-x_{i+1}y_i)(y_i+y_{i+1})}{\sum_{i}(x_i+x_{i+1})(y_{i+1}-y_i)}
`$

This is particularly useful, since if we have any linear function over $`(x,y)`$ (say, a linear gradient between two colors), the average color in the polygon would be the value of that function at the centroid!

## Evaluation of Filtered Polygons

Any polynomial-based (windowed or not) filter can be evaluated over a polygon with this approach.

A simple practical example is the tent filter (for [Bilinear filtering](https://en.wikipedia.org/wiki/Bilinear_interpolation)). It is effectively evaluating the integral $`(1-x)(1-y)=xy-x-y+1`$ over $`0\le x\le1,0\le y\le1`$, which we can now evaluate as:

$`
\frac{1}{24}\sum_{i}(x_iy_{i+1}-x_{i+1}y_i)(12-4( x_i+y_i+x_{i+1}+y_{i+1})+2(x_iy_i+x_{i+1}y_{i+1})+ 
x_iy_{i+1}+x_{i+1}y_i)
`$

