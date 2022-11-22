// Copyright 2020-2022, University of Colorado Boulder

/**
 * Sepia filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { ColorMatrixFilter, scenery } from '../imports.js';

export default class Sepia extends ColorMatrixFilter {

  private readonly amount: number;

  /**
   * @param [amount] - The amount of the effect, from 0 (none) to 1 (full sepia)
   */
  public constructor( amount = 1 ) {
    assert && assert( isFinite( amount ), 'Sepia amount should be finite' );
    assert && assert( amount >= 0, 'Sepia amount should be non-negative' );
    assert && assert( amount <= 1, 'Sepia amount should be at most 1' );

    super(
      0.393 + 0.607 * ( 1 - amount ), 0.769 - 0.769 * ( 1 - amount ), 0.189 - 0.189 * ( 1 - amount ), 0, 0,
      0.349 - 0.349 * ( 1 - amount ), 0.686 + 0.314 * ( 1 - amount ), 0.168 - 0.168 * ( 1 - amount ), 0, 0,
      0.272 - 0.272 * ( 1 - amount ), 0.534 - 0.534 * ( 1 - amount ), 0.131 + 0.869 * ( 1 - amount ), 0, 0,
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
    return `sepia(${toSVGNumber( this.amount )})`;
  }

  public override isDOMCompatible(): boolean {
    return true;
  }

  public static readonly FULL = new Sepia( 1 );
}

scenery.register( 'Sepia', Sepia );
