/**
 * Provides the source text for the uber vertex shader to be used in WebGLBlock.js.  This is an uber shader, which means
 * it contains logic for rendering different types, so that shader programs do not need to be switched as frequently
 * (which can reportedly decreased performance).
 *
 * Note that when the attributes and uniforms are changed, they must also be updated in WebGLBlock.js.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
*/

precision mediump float;

//The vertex to be transformed
attribute vec3 aVertex;

// The texture coordinate
attribute vec2 aTexCoord;

// The projection matrix
uniform mat4 uProjectionMatrix;

// The model-view matrix
uniform mat4 uModelViewMatrix;

// The texture coordinates (if any)
//TODO: Is this needed here in the vertex shader?
varying vec2 texCoord;

// The color to render (if any)
//TODO: Is this needed here in the vertex shader?
uniform vec4 uColor;

void main() {

  // Set the position for the fragments
  // The aVertex must be referenced first, since the usage here determines that it will be the 0th attribute, See #310
  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4( aVertex, 1 );

  // This texture is not needed for rectangles, but we (JO/SR) don't expect it to be expensive, so we leave it for simplicity
  texCoord = aTexCoord;
}