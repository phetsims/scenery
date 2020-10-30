// Copyright 2020, University of Colorado Boulder

/**
 * GaussianBlur filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';

class GaussianBlur extends Filter {
  /**
   * @param {number} standardDeviation
   */
  constructor( standardDeviation ) {
    assert && assert( typeof standardDeviation === 'number', 'GaussianBlur standardDeviation should be a number' );
    assert && assert( isFinite( standardDeviation ), 'GaussianBlur standardDeviation should be finite' );
    assert && assert( standardDeviation >= 0, 'GaussianBlur standardDeviation should be non-negative' );

    super();

    // @public {number}
    this.standardDeviation = standardDeviation;
  }

  /**
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `blur(${toSVGNumber( this.standardDeviation )}px)`;
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

scenery.register( 'GaussianBlur', GaussianBlur );
export default GaussianBlur;