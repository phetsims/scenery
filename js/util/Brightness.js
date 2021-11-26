// Copyright 2020-2021, University of Colorado Boulder

/**
 * Brightness filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, ColorMatrixFilter } from '../imports.js';

class Brightness extends ColorMatrixFilter {
  /**
   * @param {number} amount - How bright to be, from 0 (dark), 1 (normal), or larger values to brighten
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'Brightness amount should be a number' );
    assert && assert( isFinite( amount ), 'Brightness amount should be finite' );
    assert && assert( amount >= 0, 'Brightness amount should be non-negative' );

    super(
      amount, 0, 0, 0, 0,
      0, amount, 0, 0, 0,
      0, 0, amount, 0, 0,
      0, 0, 0, 1, 0
    );

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
    return `brightness(${toSVGNumber( this.amount )})`;
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

// @public {Brightness} - Fully darkens the content
Brightness.BLACKEN = new Brightness( 0 );

scenery.register( 'Brightness', Brightness );
export default Brightness;