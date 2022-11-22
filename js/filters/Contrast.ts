// Copyright 2020-2022, University of Colorado Boulder

/**
 * Contrast filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { ColorMatrixFilter, scenery } from '../imports.js';

export default class Contrast extends ColorMatrixFilter {

  private readonly amount: number;

  /**
   * @param amount - The amount of the effect, from 0 (gray), 1 (normal), or above for high-contrast
   */
  public constructor( amount: number ) {
    assert && assert( isFinite( amount ), 'Contrast amount should be finite' );
    assert && assert( amount >= 0, 'Contrast amount should be non-negative' );

    super(
      amount, 0, 0, 0, -( 0.5 * amount ) + 0.5,
      0, amount, 0, 0, -( 0.5 * amount ) + 0.5,
      0, 0, amount, 0, -( 0.5 * amount ) + 0.5,
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
    return `contrast(${toSVGNumber( this.amount )})`;
  }

  public override isDOMCompatible(): boolean {
    return true;
  }

  // Turns the content gray
  public static readonly GRAY = new Contrast( 0 );
}

scenery.register( 'Contrast', Contrast );
