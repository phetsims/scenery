// Copyright 2019, University of Colorado Boulder

/**
 * Represents an image with a specific center "offset". Considered immutable (with an immutable image, the Canvas if
 * provided should not change).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const scenery = require( 'SCENERY/scenery' );
  const Vector2 = require( 'DOT/Vector2' );

  let globalIdCounter = 1;

  class SpriteImage {
    /**
     * @param {HTMLImageElement|HTMLCanvasElement} image
     * @param {Vector2} offset - A 2d offset from the upper-left of the image which is considered the "center".
     */
    constructor( image, offset ) {
      assert && assert( image instanceof HTMLImageElement ||
                        image instanceof HTMLCanvasElement );
      assert && assert( offset instanceof Vector2 );

      // @public {number}
      this.id = globalIdCounter++;

      // @public {HTMLImageElement|HTMLCanvasElement}
      this.image = image;

      // @public {Vector2}
      this.offset = offset;
    }
  }

  return scenery.register( 'SpriteImage', SpriteImage );
} );
