// Copyright 2020, University of Colorado Boulder

/**
 * HueRotate filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';

class HueRotate extends Filter {
  /**
   * @param {number} amount - In degrees
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'HueRotate amount should be a number' );
    assert && assert( isFinite( amount ), 'HueRotate amount should be finite' );
    assert && assert( amount >= 0, 'HueRotate amount should be non-negative' );

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
    return `hue-rotate(${toSVGNumber( this.amount )}deg)`;
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

scenery.register( 'HueRotate', HueRotate );
export default HueRotate;