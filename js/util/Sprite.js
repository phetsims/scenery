// Copyright 2019, University of Colorado Boulder

/**
 * Represents a single sprite for the Sprites node, whose image can change over time (if it gets regenerated, etc.).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const Property = require( 'AXON/Property' );
  const scenery = require( 'SCENERY/scenery' );
  const SpriteImage = require( 'SCENERY/util/SpriteImage' );

  class Sprite {
    /**
     * @param {SpriteImage} spriteImage - The initial SpriteImage
     */
    constructor( spriteImage ) {
      assert && assert( spriteImage instanceof SpriteImage );

      // @public {Property.<SpriteImage>}
      this.imageProperty = new Property( spriteImage );
    }
  }

  return scenery.register( 'Sprite', Sprite );
} );
