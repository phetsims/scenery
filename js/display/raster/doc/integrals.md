
TODO: flesh this out, and document! For now, it's a place where dumped formulas will be

Arbitrary polynomial:

$`\iint_Px^my^n\,dxdy=\frac{m!n!}{(m+n+2)!}\sum_{i}\left[ (x_iy_{i+1}-x_{i+1}y_i) \sum_{p=0}^m\sum_{q=0}^n \binom{p+q}{q}\binom{m+n-p-q}{n-q}x_i^{m-p}x_{i+1}^py_i^{n-q}y_{i+1}^q \right]`$

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

Notable simplifications:

For $`x^0y^0=1`$, we'll have the [Shoelace formula](https://en.wikipedia.org/wiki/Shoelace_formula) for finding the area of a polygon:

$`
area_P=\iint_P1\,dxdy=
\frac{1}{2}
\sum_{i}
(x_i+x_{i+1})(y_{i+1}-y_i)
`$

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

The tent filter (for [Bilinear filtering](https://en.wikipedia.org/wiki/Bilinear_interpolation)) is 
