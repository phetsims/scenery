// Copyright 2020-2022, University of Colorado Boulder

/**
 * Opacity filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { CanvasContextWrapper, Filter, scenery } from '../imports.js';

export default class Opacity extends Filter {

  private readonly amount: number;

  /**
   * NOTE: Generally prefer setting a Node's opacity, unless this is required for stacking of filters.
   *
   * @param amount - The amount of opacity, from 0 (invisible) to 1 (fully visible)
   */
  public constructor( amount: number ) {
    assert && assert( isFinite( amount ), 'Opacity amount should be finite' );
    assert && assert( amount >= 0, 'Opacity amount should be non-negative' );
    assert && assert( amount <= 1, 'Opacity amount should be no greater than 1' );

    super();

    this.amount = amount;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   */
  public getCSSFilterString(): string {
    return `opacity(${toSVGNumber( this.amount )})`;
  }

  public override isDOMCompatible(): boolean {
    return true;
  }

  public applyCanvasFilter( wrapper: CanvasContextWrapper ): void {
    throw new Error( 'unimplemented' );
  }

  public applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    throw new Error( 'unimplemented' );
  }
}

scenery.register( 'Opacity', Opacity );
