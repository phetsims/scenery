// Copyright 2020-2022, University of Colorado Boulder

/**
 * GaussianBlur filter
 *
 * EXPERIMENTAL! DO not use in production code yet
 *
 * TODO: preventFit OR handle bounds increase (or both)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import { CanvasContextWrapper, Filter, scenery, svgns } from '../imports.js';

export default class GaussianBlur extends Filter {

  private readonly standardDeviation: number;

  /**
   * @param standardDeviation
   * @param [filterRegionPercentage]
   */
  public constructor( standardDeviation: number, filterRegionPercentage = 15 ) {
    assert && assert( isFinite( standardDeviation ), 'GaussianBlur standardDeviation should be finite' );
    assert && assert( standardDeviation >= 0, 'GaussianBlur standardDeviation should be non-negative' );

    super();

    this.standardDeviation = standardDeviation;

    this.filterRegionPercentageIncrease = filterRegionPercentage;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   */
  public getCSSFilterString(): string {
    return `blur(${toSVGNumber( this.standardDeviation )}px)`;
  }

  /**
   * Appends filter sub-elements into the SVG filter element provided. Should include an in=${inName} for all inputs,
   * and should either output using the resultName (or if not provided, the last element appended should be the output).
   * This effectively mutates the provided filter object, and will be successively called on all Filters to build an
   * SVG filter object.
   */
  public applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    // e.g. <feGaussianBlur stdDeviation="[radius radius]" edgeMode="[edge mode]" >
    const feGaussianBlur = document.createElementNS( svgns, 'feGaussianBlur' );
    feGaussianBlur.setAttribute( 'stdDeviation', toSVGNumber( this.standardDeviation ) );
    feGaussianBlur.setAttribute( 'edgeMode', 'none' ); // Don't pad things!
    svgFilter.appendChild( feGaussianBlur );

    feGaussianBlur.setAttribute( 'in', inName );
    if ( resultName ) {
      feGaussianBlur.setAttribute( 'result', resultName );
    }
    svgFilter.appendChild( feGaussianBlur );
  }

  public override isDOMCompatible(): boolean {
    return true;
  }

  public override isSVGCompatible(): boolean {
    return true;
  }

  public applyCanvasFilter( wrapper: CanvasContextWrapper ): void {
    throw new Error( 'unimplemented' );
  }
}

scenery.register( 'GaussianBlur', GaussianBlur );
