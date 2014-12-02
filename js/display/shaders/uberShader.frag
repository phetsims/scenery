/**
 * Provides the source text for the uber vertex shader to be used in WebGLBlock.js.  This is an uber shader, which means
 * it contains logic for rendering different types, so that shader programs do not need to be switched as frequently
 * (which can reportedly decreased performance).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
*/

//Directive to indicate high precision
precision mediump float;

//Texture coordinates (for images)
varying vec2 texCoord;

//Color (rgba) for filled items
uniform vec4 uColor;

//Fragment type such as fragmentTypeFill or fragmentTypeTexture
uniform int uFragmentType;

//Texture (if any)
uniform sampler2D uTexture;

void main() {

  // Check for fragmentTypeFill (0)
  if (uFragmentType==0){
    gl_FragColor = uColor;
  }else if (uFragmentType==1){
    gl_FragColor = texture2D( uTexture, texCoord );
  }
}