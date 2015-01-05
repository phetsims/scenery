//  Copyright 2002-2014, University of Colorado Boulder

/**
 * A single image with different regions within the image representing different distinct textures to be drawn in WebGL.
 * The TextureRenderer will normally use more than one SpriteSheet for rendering.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  /**
   *
   * @constructor
   */
  function SpriteSheet() {
    this.image = document.createElement( 'canvas' );

    // Use the max supported texture size (according to http://codeflow.org/entries/2013/feb/22/how-to-write-portable-webgl/ )
    this.image.width = 2048;
    this.image.height = 2048;
    this.context = this.image.getContext( '2d' );

    // Flag as dirty initially because it has not yet been registered with the GPU as a texture unit.
    // @public, settable
    this.dirty = true;
  }

  return inherit( Object, SpriteSheet, {
    addImage: function( image ) {
      this.context.drawImage( image, 0, 0 );
      this.dirty = true;
    }
  } );
} );