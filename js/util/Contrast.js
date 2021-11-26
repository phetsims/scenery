// Copyright 2020-2021, University of Colorado Boulder

/**
 * Contrast filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, ColorMatrixFilter } from '../imports.js';

class Contrast extends ColorMatrixFilter {
  /**
   * @param {number} amount - The amount of the effect, from 0 (gray), 1 (normal), or above for high-contrast
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
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
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

// @public {Contrast} - Turns the content gray
Contrast.GRAY = new Contrast( 0 );

scenery.register( 'Contrast', Contrast );
export default Contrast;