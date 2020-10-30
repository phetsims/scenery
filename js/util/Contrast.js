// Copyright 2020, University of Colorado Boulder

/**
 * Contrast filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import ColorMatrixFilter from './ColorMatrixFilter.js';

class Contrast extends ColorMatrixFilter {
  /**
   * @param {number} amount
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'Contrast amount should be a number' );
    assert && assert( isFinite( amount ), 'Contrast amount should be finite' );
    assert && assert( amount >= 0, 'Contrast amount should be non-negative' );

    super(
      amount, 0, 0, 0, -( 0.5 * amount ) + 0.5,
      0, amount, 0, 0, -( 0.5 * amount ) + 0.5,
      0, 0, amount, 0, -( 0.5 * amount ) + 0.5,
      0, 0, 0, 1, 0
    );

    // @public {number}
    this.amount = amount;
  }

  /**
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `contrast(${toSVGNumber( this.amount )})`;
  }

  /**
   * @public
   * @override
   *
   * @returns {*}
   */
  isDOMCompatible() {
    return true;
  }
}

scenery.register( 'Contrast', Contrast );
export default Contrast;