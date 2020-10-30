// Copyright 2020, University of Colorado Boulder

/**
 * Saturate filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import ColorMatrixFilter from './ColorMatrixFilter.js';

class Saturate extends ColorMatrixFilter {
  /**
   * @param {number} amount
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'Saturate amount should be a number' );
    assert && assert( isFinite( amount ), 'Saturate amount should be finite' );
    assert && assert( amount >= 0, 'Saturate amount should be non-negative' );

    // near https://drafts.fxtf.org/filter-effects/#attr-valuedef-type-huerotate
    super(
      0.213 + 0.787 * amount, 0.715 - 0.715 * amount, 0.072 - 0.072 * amount, 0, 0,
      0.213 - 0.213 * amount, 0.715 - 0.285 * amount, 0.072 - 0.072 * amount, 0, 0,
      0.213 - 0.213 * amount, 0.715 - 0.715 * amount, 0.072 - 0.928 * amount, 0, 0,
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
    return `saturate(${toSVGNumber( this.amount )})`;
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

scenery.register( 'Saturate', Saturate );
export default Saturate;