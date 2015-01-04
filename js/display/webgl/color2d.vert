//This shader assumes 2d vertices that have a specified color.

// Position
attribute vec3 aPosition;

// Color for the vertex
attribute vec4 aVertexColor;

// Output to the fragment shader
varying vec4 vColor;

// (x,y) size of the viewport
uniform vec2 uResolution;

// Components of the affine transform matrix.  6 float elements, so specified as two vec3
attribute vec3 aTransform1;
attribute vec3 aTransform2;

void main(void) {

  //This transform code is based on http://www.html5rocks.com/en/tutorials/webgl/webgl_fundamentals/
  //TODO: Should be converted to matrix multiply, probably faster.

  // Just do the affine transform ourselves.
  // see http://cs.iupui.edu/~sfang/cs550/cs550-note3.pdf
  vec2 transformed = vec2( aTransform1.x * aPosition.x + aTransform1.y * aPosition.y + aTransform1.z,
                           aTransform2.x * aPosition.x + aTransform2.y * aPosition.y + aTransform2.z );

  // convert the rectangle from pixels to 0.0 to 1.0
  vec2 zeroToOne = vec2( transformed.x / uResolution.x , transformed.y / uResolution.y );

  // convert from 0->1 to 0->2
  vec2 zeroToTwo = zeroToOne * 2.0;

  // convert from 0->2 to -1->+1 (clipspace)
  vec2 clipSpace = zeroToTwo - 1.0;

  gl_Position = vec4(clipSpace * vec2(1, -1), aPosition.z, 1.0); //0. is the z, and 1 is w
  vColor = aVertexColor;
}