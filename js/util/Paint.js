// Copyright 2014-2020, University of Colorado Boulder

/**
 * Base type for gradients and patterns (and NOT the only type for fills/strokes)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';

let globalId = 1;

class Paint {
  constructor() {
    // @public (scenery-internal) {string}
    this.id = `paint${globalId++}`;

    // @protected {Matrix3|null}
    this.transformMatrix = null;
  }

  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   * @public
   *
   * @returns {*}
   */
  getCanvasStyle() {
    throw new Error( 'abstract method' );
  }

  /**
   * Sets how this paint (pattern/gradient) is transformed, compared with the local coordinate frame of where it is
   * used.
   * @public
   *
   * NOTE: This should only be used before the pattern/gradient is ever displayed.
   * TODO: Catch if this is violated?
   *
   * @param {Matrix3} transformMatrix
   * @returns {Paint} - for chaining
   */
  setTransformMatrix( transformMatrix ) {
    if ( this.transformMatrix !== transformMatrix ) {
      this.transformMatrix = transformMatrix;
    }
    return this;
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return this.id;
  }
}

// @public {boolean}
Paint.prototype.isPaint = true;

scenery.register( 'Paint', Paint );
export default Paint;