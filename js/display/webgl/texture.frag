// From http://www.html5rocks.com/en/tutorials/webgl/webgl_fundamentals/

precision mediump float;

// our texture
uniform sampler2D uImage;

// the texCoords passed in from the vertex shader.
varying vec2 vTextureCoordinate;

void main() {
   gl_FragColor = texture2D(uImage, vTextureCoordinate);
}