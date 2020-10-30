// Copyright 2020, University of Colorado Boulder

/**
 * Grayscale filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';

class Grayscale extends Filter {
  /**
   * @param {number} amount
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'Grayscale amount should be a number' );
    assert && assert( isFinite( amount ), 'Grayscale amount should be finite' );
    assert && assert( amount >= 0, 'Grayscale amount should be non-negative' );
    assert && assert( amount <= 1, 'Grayscale amount should be no greater than 1' );

    super();

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
    return `grayscale(${toSVGNumber( this.amount )})`;
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

scenery.register( 'Grayscale', Grayscale );
export default Grayscale;