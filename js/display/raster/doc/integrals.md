# Integrals over Polygons

## Green's Theorem and Polygons

Using [Green's Theorem](https://en.wikipedia.org/wiki/Green%27s_theorem), we can convert a double integral over a region into a line integral over the (closed, oriented counterclockwise) boundary of the region:

For curves parameterized on $`t`$:

$`
\oint_P\left(L\,\frac{dx}{dt}+M\,\frac{dy}{dt}\right)dt=\iint_P \left( \frac{\partial M}{\partial x}-\frac{\partial L}{\partial y} \right)\,dx\,dy
`$

For polygons, this means that if we can evaluate a line integral over each edge (point $`(x_i,y_i)`$ to point $`(x_{i+1},y_{i+1})`$), we can sum up each edge's contribution to get the double integral over the polygon. Notably our line segments are parameterized:

$`
x(t)=(1-t)x_i+(t)x_{i+1}=x_i+t(x_{i+1}-x_i)
`$

$`
y(t)=(1-t)y_i+(t)y_{i+1}=y_i+t(y_{i+1}-y_i)
`$

with the derivatives:

$`
x'(t)=x_{i+1}-x_i
`$

$`
y'(t)=y_{i+1}-y_i
`$

There are two notable observations:

1. If we reverse an edge (swap its endpoints), it will swap the sign of the contribution to the integral.
2. We are evaluating this on closed polygons, so any terms that only depend on one endpoint will cancel out (e.g. $`x_i^2y_i`$ and $`-x_{i+1}^2y_{i+1}`$ in summations will cancel out, and those terms will always be the additive inverse of each other$)

This means that polygons with holes can be evaluated by visiting the holes with the opposite orientation (clockwise).

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

$`\iint_Px^my^n\,dx\,dy=\frac{m!n!}{(m+n+2)!}\sum_{i}\left[ (x_iy_{i+1}-x_{i+1}y_i) \sum_{p=0}^m\sum_{q=0}^n \binom{p+q}{q}\binom{m+n-p-q}{n-q}x_i^{m-p}x_{i+1}^py_i^{n-q}y_{i+1}^q \right]`$

(Conjecture, matches Mathematica output precisely). The contributions of each term can be summed up individually to integrate arbitrary polynomials.

e.g. for $`x^4y^2`$ in matrix form:

$`
\iint_Px^4y^2\,dx\,dy=
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
area_P=\iint_P1\,dx\,dy=
\frac{1}{2}
\sum_{i}
(x_i+x_{i+1})(y_{i+1}-y_i)
`$

## Centroids of Polygons

For $`x`$ and $`y`$, we have:

$`
\iint_Px\,dx\,dy=
\frac{1}{6}
\sum_{i}
(x_iy_{i+1}-x_{i+1}y_i)(x_i+x_{i+1})
`$

$`
\iint_Py\,dx\,dy=
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

## Evaluation of Distance over Polygons

Above, we saw the centroid is useful to compute exact linear gradient contributions. For purely-circular radial gradients, the equivalent is also possible to compute! We'll need to instead integrate $`r=\sqrt{x^2+y^2}`$ (we can determine the average by dividing by the area).

We'll need to transform to polar coordinates first:

$`r=\sqrt{x^2+y^2}`$

$`\theta=\tan^{-1}\frac{y}{x}`$

We'll want to evaluate with Green's Theorem in polar coordinates:

$`
\oint_P\left(L\,\frac{dr}{dt}+M\,\frac{d\theta}{dt}\right)dt=\iint_P \left( \frac{\partial M}{\partial r}-\frac{\partial L}{\partial \theta} \right)\,dA
`$

but we'll want to evaluate $`r^2`$ due to the coordinate change.

If we pick $`M=\frac{1}{3}r^3`$ and $`L=0`$, the double integral will be our desired integral (note, $`M=\frac{1}{2}r^2`$ gives us the same Shoelace-like area formula).

Given our definitions of $`x=x_i+t(x_{i+1}-x_i)`$ and $`y=y_i+t(y_{i+1}-y_i)`$:

$`\frac{d\theta}{dt}=\frac{d}{dt}\tan^{-1}\frac{y}{x}=\frac{d}{dt}\tan^{-1}\frac{y_i+t(y_{i+1}-y_i)}{x_i+t(x_{i+1}-x_i)}=\frac{x_iy_{i+1}-x_{i+1}y_i}{t^2((x_{i_1}-x_i)^2+(y_{i+1}-y_i)^2)-2t(x_i^2-x_ix_{i+1}-y_iy_{i+1}+y_i^2)+(x_i^2+y_i^2)}`$

Thus given $`M`$ and $`\frac{d\theta}{dt}`$, we can evaluate (with Mathematica in this case):

$`
\oint_P\left(M\,\frac{d\theta}{dt}\right)dt=\int_0^1\frac{1}{3}r^3\frac{d\theta}{dt}\,dt=
\frac{s}{6d_{xy}^3}\left[
  d_{xy}\left( q_0( x_i^2 - x_ix_{i+1} - y_id_y ) + q_1( k_x + y_{i+1}d_y ) \right) +
  s^2\log\frac{k_x + k_y + d_{xy}q_1}{x_id_x + q_0d_{xy} + y_id_y}
\right]
`$

with

$`d_x = x_{i+1} - x_i`$

$`d_y = y_{i+1} - y_i`$

$`s = x_iy_{i+1} - y_ix_{i+1}`$

$`d_{xy} = \sqrt{d_xd_x + d_yd_y}`$

$`q_0 = \sqrt{x_ix_i + y_iy_i}`$

$`q_1 = \sqrt{x_{i+1}x_{i+1} + y_{i+1}y_{i+1}}`$

$`k_x = x_{i+1}x_{i+1} - x_ix_{i+1}`$

$`k_y = y_{i+1}y_{i+1} - y_iy_{i+1}`$

thus

$`
\iint_P\sqrt{x^2+y^2}\,dx\,dy=
\frac{s}{6d_{xy}^3}\left[
  d_{xy}\left( q_0( x_i^2 - x_ix_{i+1} - y_id_y ) + q_1( k_x + y_{i+1}d_y ) \right) +
  s^2\log\frac{k_x + k_y + d_{xy}q_1}{x_id_x + q_0d_{xy} + y_id_y}
\right]
`$

## Checking if a Polygon is not Closed

We can integrate $`0`$ over a polygon's edges in a similar way, to compute if the polygon is not closed (there are some cases where this happens and is useful).

$`0=\sum_{i}(x_iy_i-x_{i+1}y_{i+1})`$

This test (and any other tests of this type) will have false-negatives (for instance, if all the points are on an x or y axis, this formula won't detect non-closed polygons). However the useful probability of that happening can be reduced by using a random point as a translation:

$`0=\sum_{i}\left[(x_i-p_x)(y_i-p_y)-(x_{i+1}-p_x)(y_{i+1}-p_y)\right]`$
