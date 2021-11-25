// Copyright 2020, University of Colorado Boulder

/**
 * Invert filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, Filter } from '../imports.js';

class Invert extends Filter {
  /**
   * @param {number} [amount] - The amount of the effect, from 0 (none) to 1 (full)
   */
  constructor( amount = 1 ) {
    assert && assert( typeof amount === 'number', 'Invert amount should be a number' );
    assert && assert( isFinite( amount ), 'Invert amount should be finite' );
    assert && assert( amount >= 0, 'Invert amount should be non-negative' );
    assert && assert( amount <= 1, 'Invert amount should be no greater than 1' );

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
    return `invert(${toSVGNumber( this.amount )})`;
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

// @public {Invert}
Invert.FULL = new Invert( 1 );

scenery.register( 'Invert', Invert );
export default Invert;