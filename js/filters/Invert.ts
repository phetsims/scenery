// Copyright 2020-2025, University of Colorado Boulder

/**
 * Invert filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/util/toSVGNumber.js';
import CanvasContextWrapper from '../util/CanvasContextWrapper.js';
import Filter from '../filters/Filter.js';
import scenery from '../scenery.js';

export default class Invert extends Filter {

  private readonly amount: number;

  /**
   * @param [amount] - The amount of the effect, from 0 (none) to 1 (full)
   */
  public constructor( amount = 1 ) {
    assert && assert( isFinite( amount ), 'Invert amount should be finite' );
    assert && assert( amount >= 0, 'Invert amount should be non-negative' );
    assert && assert( amount <= 1, 'Invert amount should be no greater than 1' );

    super();

    this.amount = amount;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   */
  public getCSSFilterString(): string {
    return `invert(${toSVGNumber( this.amount )})`;
  }

  public override isDOMCompatible(): boolean {
    return true;
  }

  public static readonly FULL = new Invert( 1 );

  public applyCanvasFilter( wrapper: CanvasContextWrapper ): void {
    throw new Error( 'unimplemented' );
  }

  public applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    throw new Error( 'unimplemented' );
  }
}

scenery.register( 'Invert', Invert );