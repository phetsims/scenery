// Copyright 2020-2022, University of Colorado Boulder

/**
 * DropShadow filter
 *
 * EXPERIMENTAL! DO not use in production code yet
 *
 * TODO: preventFit OR handle bounds increase (or both)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import Vector2 from '../../../dot/js/Vector2.js';
import { ColorDef, Filter, TColor, PaintDef, scenery } from '../imports.js';

export default class DropShadow extends Filter {

  private readonly offset: Vector2;
  private readonly blurRadius: number;
  private readonly color: TColor;
  private readonly colorCSS: string;

  /**
   * @param offset
   * @param blurRadius
   * @param color
   * @param [filterRegionPercentage]
   */
  public constructor( offset: Vector2, blurRadius: number, color: TColor, filterRegionPercentage = 15 ) {
    assert && assert( offset.isFinite(), 'DropShadow offset should be finite' );
    assert && assert( isFinite( blurRadius ), 'DropShadow blurRadius should be finite' );
    assert && assert( blurRadius >= 0, 'DropShadow blurRadius should be non-negative' );
    assert && assert( ColorDef.isColorDef( color ), 'DropShadow color should be a ColorDef' );

    super();

    // TODO: consider linking to the ColorDef (if it's a Property), or indicating that we need an update

    this.offset = offset;
    this.blurRadius = blurRadius;
    this.color = color;
    this.colorCSS = PaintDef.toColor( color ).toCSS();

    this.filterRegionPercentageIncrease = filterRegionPercentage;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   */
  public getCSSFilterString(): string {
    return `drop-shadow(${toSVGNumber( this.offset.x )}px ${toSVGNumber( this.offset.y )}px ${toSVGNumber( this.blurRadius )}px ${this.colorCSS})`;
  }

  public override isDOMCompatible(): boolean {
    return true;
  }

  public applyCanvasFilter(): void {
    throw new Error( 'unimplemented' );
  }

  public applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    throw new Error( 'unimplemented' );
  }
}

scenery.register( 'DropShadow', DropShadow );
