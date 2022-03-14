// Copyright 2020-2022, University of Colorado Boulder

/**
 * Invert filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { scenery, Filter, CanvasContextWrapper } from '../imports.js';

export default class Invert extends Filter {

  amount: number;

  /**
   * @param [amount] - The amount of the effect, from 0 (none) to 1 (full)
   */
  constructor( amount: number = 1 ) {
    assert && assert( typeof amount === 'number', 'Invert amount should be a number' );
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
  getCSSFilterString(): string {
    return `invert(${toSVGNumber( this.amount )})`;
  }

  isDOMCompatible() {
    return true;
  }

  static FULL: Invert;

  applyCanvasFilter( wrapper: CanvasContextWrapper ): void {
    throw new Error( 'unimplemented' );
  }

  applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    throw new Error( 'unimplemented' );
  }
}

Invert.FULL = new Invert( 1 );

scenery.register( 'Invert', Invert );
