//This shader assumes 2d vertices that have a specified color.

attribute vec2 aPosition;
attribute vec4 aVertexColor;
varying vec4 vColor;
void main(void) { //pre-built function
 gl_Position = vec4(aPosition, 0., 1.); //0. is the z, and 1 is w
 vColor = aVertexColor;
}