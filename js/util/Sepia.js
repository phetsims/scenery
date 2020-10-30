// Copyright 2020, University of Colorado Boulder

/**
 * Sepia filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';

class Sepia extends Filter {
  /**
   * @param {number} [amount]
   */
  constructor( amount = 1 ) {
    assert && assert( typeof amount === 'number', 'Sepia amount should be a number' );
    assert && assert( isFinite( amount ), 'Sepia amount should be finite' );
    assert && assert( amount >= 0, 'Sepia amount should be non-negative' );
    assert && assert( amount <= 1, 'Sepia amount should be at most 1' );

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
    return `sepia(${toSVGNumber( this.amount )})`;
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

scenery.register( 'Sepia', Sepia );
export default Sepia;