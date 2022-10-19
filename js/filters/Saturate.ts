// Copyright 2020-2022, University of Colorado Boulder

/**
 * Saturate filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { ColorMatrixFilter, scenery } from '../imports.js';

export default class Saturate extends ColorMatrixFilter {

  private readonly amount: number;

  /**
   * @param amount - The amount of the effect, from 0 (no saturation), 1 (normal), or higher to over-saturate
   */
  public constructor( amount: number ) {
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
  public override getCSSFilterString(): string {
    return `saturate(${toSVGNumber( this.amount )})`;
  }

  public override isDOMCompatible(): boolean {
    return true;
  }
}

scenery.register( 'Saturate', Saturate );
