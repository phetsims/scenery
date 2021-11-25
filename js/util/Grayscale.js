// Copyright 2020-2021, University of Colorado Boulder

/**
 * Grayscale filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, ColorMatrixFilter } from '../imports.js';

class Grayscale extends ColorMatrixFilter {
  /**
   * @param {number} [amount] - The amount of the effect, from 0 (none) to 1 (full)
   */
  constructor( amount = 1 ) {
    assert && assert( typeof amount === 'number', 'Grayscale amount should be a number' );
    assert && assert( isFinite( amount ), 'Grayscale amount should be finite' );
    assert && assert( amount >= 0, 'Grayscale amount should be non-negative' );
    assert && assert( amount <= 1, 'Grayscale amount should be no greater than 1' );

    const n = 1 - amount;

    // https://drafts.fxtf.org/filter-effects/#grayscaleEquivalent
    // (0.2126 + 0.7874 * [1 - amount]) (0.7152 - 0.7152  * [1 - amount]) (0.0722 - 0.0722 * [1 - amount]) 0 0
    // (0.2126 - 0.2126 * [1 - amount]) (0.7152 + 0.2848  * [1 - amount]) (0.0722 - 0.0722 * [1 - amount]) 0 0
    // (0.2126 - 0.2126 * [1 - amount]) (0.7152 - 0.7152  * [1 - amount]) (0.0722 + 0.9278 * [1 - amount]) 0 0
    // 0 0 0 1 0
    super(
      0.2126 + 0.7874 * n, 0.7152 - 0.7152 * n, 0.0722 - 0.0722 * n, 0, 0,
      0.2126 - 0.2126 * n, 0.7152 + 0.2848 * n, 0.0722 - 0.0722 * n, 0, 0,
      0.2126 - 0.2126 * n, 0.7152 - 0.7152 * n, 0.0722 + 0.9278 * n, 0, 0,
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

// @public {Grayscale}
Grayscale.FULL = new Grayscale( 1 );

scenery.register( 'Grayscale', Grayscale );
export default Grayscale;