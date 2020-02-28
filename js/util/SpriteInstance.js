// Copyright 2019-2020, University of Colorado Boulder

/**
 * Represents a single instance on the screen of a given Sprite object. It can have its own transformation matrix, and
 * is set up as a lightweight container of information, for high-performance usage (with pooling).
 *
 * Its individual parameters should generally be mutated directly by the client.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import scenery from '../scenery.js';

class SpriteInstance {
  constructor() {

    // @public {Sprite|null} - This should be set to a `Sprite` object which is the sprite that should be displayed.
    this.sprite = null;

    // @public {Matrix3} - Please just mutate the given Matrix3 for performance. If the matrix represents something
    // other than just a translation, please set the `isTranslation` flag to false.
    this.matrix = new Matrix3().setToAffine( 1, 0, 0, 0, 1, 0 ); // initialized to trigger the affine flag

    // @public {boolean} - Whether the transformation is just a translation (if true, can be faster for Canvas).
    this.isTranslation = true;

    // @public {number} - The general opacity/alpha of the displayed sprite (see Node's opacity)
    this.alpha = 1;
  }

  /**
   * @private - For pooling. Please use SpriteInstance.dirtyFromPool() to grab a copy
   */
  initialize() {
    // We need an empty initialization method here, so that we can grab dirty versions and use them for higher
    // performance.
  }
}

Poolable.mixInto( SpriteInstance, {
  maxSize: 1000,
  initialize: SpriteInstance.prototype.initialize
} );

scenery.register( 'SpriteInstance', SpriteInstance );
export default SpriteInstance;