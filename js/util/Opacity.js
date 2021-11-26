// Copyright 2020-2021, University of Colorado Boulder

/**
 * Opacity filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, Filter } from '../imports.js';

class Opacity extends Filter {
  /**
   * NOTE: Generally prefer setting a Node's opacity, unless this is required for stacking of filters.
   *
   * @param {number} amount - The amount of opacity, from 0 (invisible) to 1 (fully visible)
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