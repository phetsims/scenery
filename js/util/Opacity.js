// Copyright 2020, University of Colorado Boulder

/**
 * Opacity filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import Filter from './Filter.js';

class Opacity extends Filter {
  /**
   * @param {number} amount
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'Opacity amount should be a number' );
    assert && assert( isFinite( amount ), 'Opacity amount should be finite' );
    assert && assert( amount >= 0, 'Opacity amount should be non-negative' );
    assert && assert( amount <= 1, 'Opacity amount should be no greater than 1' );

    super();

    // @public {number}
    this.amount = amount;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `opacity(${toSVGNumber( this.amount )})`;
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

scenery.register( 'Opacity', Opacity );
export default Opacity;