// Copyright 2019-2021, University of Colorado Boulder

/**
 * Represents an image with a specific center "offset". Considered immutable (with an immutable image, the Canvas if
 * provided should not change).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Shape from '../../../kite/js/Shape.js';
import merge from '../../../phet-core/js/merge.js';
import Imageable from '../nodes/Imageable.js';
import scenery from '../scenery.js';

let globalIdCounter = 1;
const scratchVector = new Vector2( 0, 0 );

class SpriteImage extends Imageable( Object ) {
  /**
   * @param {string|HTMLImageElement|HTMLCanvasElement|Array} image
   * @param {Vector2} offset - A 2d offset from the upper-left of the image which is considered the "center".
   * @param {Object} [options]
   */
  constructor( image, offset, options ) {
    assert && assert( image instanceof HTMLImageElement || image instanceof HTMLCanvasElement );
    assert && assert( offset instanceof Vector2 );

    options = merge( {
      hitTestPixels: false,
      pickable: true,
      image: image
    }, options );

    super();

    // @public (read-only) {number}
    this.id = globalIdCounter++;

    // @public (read-only) {Vector2}
    this.offset = offset;

    // @public (read-only) {boolean}
    this.pickable = options.pickable;

    // @private {Shape|null} - lazily constructed
    this.shape = null;

    // Initialize Imageable items (including the image itself)
    this.setImage( image );
    Object.keys( Imageable.DEFAULT_OPTIONS ).forEach( name => {
      if ( options[ name ] !== undefined ) {
        this[ name ] = options[ name ];
      }
    } );
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get width() {
    return this.imageWidth;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get height() {
    return this.imageHeight;
  }

  /**
   * Returns a Shape that represents the hit-testable area of this SpriteImage.
   * @public
   *
   * @returns {Shape}
   */
  getShape() {
    if ( !this.pickable ) {
      return new Shape();
    }

    if ( !this.shape ) {
      if ( this.hitTestPixels ) {
        this.ensureImageData();
        if ( this.imageData ) {
          this.shape = Imageable.hitTestDataToShape( this.imageData, this.width, this.height );
        }
        else {
          // Empty, if we haven't been able to load image data (even if we have a width/height)
          return new Shape();
        }
      }
      else if ( this.width && this.height ) {
        this.shape = Shape.rect( 0, 0, this.width, this.height );
      }
      else {
        // If we have no width/height
        return new Shape();
      }

      // Apply our offset
      this.shape = this.shape.transformed( Matrix3.translation( -this.offset.x, -this.offset.y ) );
    }

    return this.shape;
  }

  /**
   * Ensures we have a computed imageData (computes it lazily if necessary).
   * @private
   */
  ensureImageData() {
    if ( !this.imageData && this.width && this.height ) {
      this.imageData = Imageable.getHitTestData( this.image, this.width, this.height );
    }
  }

  /**
   * Returns whether a given point is considered "inside" the SpriteImage.
   * @public
   *
   * @param {Vector2} point
   * @returns {boolean}
   */
  containsPoint( point ) {

    if ( !this.pickable ) {
      return false;
    }

    const width = this.width;
    const height = this.height;

    // If our image isn't really loaded yet, bail out
    if ( !width && !height ) {
      return false;
    }

    const position = scratchVector.set( point ).add( this.offset );

    // Initial position check (are we within the rectangle)
    if ( position.x < 0 || position.y < 0 || position.x > width || position.y > height ) {
      return false;
    }

    if ( !this.hitTestPixels ) {
      return true;
    }
    else {
      // Lazy-load image data
      this.ensureImageData();

      // And test if it's available
      if ( this.imageData ) {
        return Imageable.testHitTestData( this.imageData, width, height, position );
      }
      else {
        return false;
      }
    }
  }
}

scenery.register( 'SpriteImage', SpriteImage );
export default SpriteImage;