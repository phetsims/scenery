// Simple fragment shader that simply uses the color provided by the vertex shader
precision mediump float;

// Color from the vertex shader
varying vec4 vColor;

// Returns the color from the vertex shader
void main(void) {
  gl_FragColor = vColor;
}