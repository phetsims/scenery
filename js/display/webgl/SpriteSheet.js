// Copyright 2002-2014, University of Colorado Boulder

/**
 * A single image with different regions within the image representing different distinct textures to be drawn in WebGL.
 * The TextureRenderer will normally use more than one SpriteSheet for rendering.
 *
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Packer = require( 'SCENERY/display/webgl/Packer' );
  var Bounds2 = require( 'DOT/Bounds2' );

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

    this.packer = new Packer( this.image.width, this.image.height );

    // Flag as dirty initially because it has not yet been registered with the GPU as a texture unit.
    // @public, settable
    this.dirty = true;
  }

  return inherit( Object, SpriteSheet, {
    /**
     * Draws the given image at a position calculated by the packer and returns the normalized bounded region
     * reserved for the image within this SpriteSheet. returns null if the image cannot be drawn, upon which a
     * new SpriteSheet will be created by SpriteBatch. see 'reserveImage' on SpriteBatch
     *
     * @param image
     * @returns {Bounds2} // in normalized coordinates
     */
    reserveImageSpace: function( image ) {
      var startPosition = this.packer.reserveSpace( image.width, image.height );
      var normalizedBounds = null;
      this.dirty = true;
      if ( startPosition ) {
        //Draw Image at specific position
        this.context.drawImage( image, startPosition.x, startPosition.y );
        normalizedBounds = Bounds2.rect( startPosition.x / this.image.width, startPosition.y / this.image.height,
          image.width / this.image.width, image.height / this.image.height );
      }
      return normalizedBounds;
    }
  } );
} );