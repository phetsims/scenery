// Copyright 2020-2021, University of Colorado Boulder

/**
 * Saturate filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, ColorMatrixFilter } from '../imports.js';

class Saturate extends ColorMatrixFilter {

  amount: number;

  /**
   * @param amount - The amount of the effect, from 0 (no saturation), 1 (normal), or higher to over-saturate
   */
  constructor( amount: number ) {
    assert && assert( typeof amount === 'number', 'Saturate amount should be a number' );
    assert && assert( isFinite( amount ), 'Saturate amount should be finite' );
    assert && assert( amount >= 0, 'Saturate amount should be non-negative' );

    // near https://drafts.fxtf.org/filter-effects/#attr-valuedef-type-huerotate
    super(
      0.213 + 0.787 * amount, 0.715 - 0.715 * amount, 0.072 - 0.072 * amount, 0, 0,
      0.213 - 0.213 * amount, 0.715 - 0.285 * amount, 0.072 - 0.072 * amount, 0, 0,
      0.213 - 0.213 * amount, 0.715 - 0.715 * amount, 0.072 - 0.928 * amount, 0, 0,
      0, 0, 0, 1, 0
    );

    this.amount = amount;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   */
  getCSSFilterString(): string {
    return `saturate(${toSVGNumber( this.amount )})`;
  }

  isDOMCompatible() {
    return true;
  }
}

scenery.register( 'Saturate', Saturate );
export default Saturate;