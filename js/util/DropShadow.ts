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
import { scenery, ColorDef, Filter, PaintDef, IColor, CanvasContextWrapper } from '../imports.js';

class DropShadow extends Filter {

  offset: Vector2;
  blurRadius: number;
  color: IColor;
  private colorCSS: string;

  /**
   * @param offset
   * @param blurRadius
   * @param color
   * @param [filterRegionPercentage]
   */
  constructor( offset: Vector2, blurRadius: number, color: IColor, filterRegionPercentage: number = 15 ) {
    assert && assert( offset instanceof Vector2, 'DropShadow offset should be a Vector2' );
    assert && assert( offset.isFinite(), 'DropShadow offset should be finite' );
    assert && assert( typeof blurRadius === 'number', 'DropShadow blurRadius should be a number' );
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
  getCSSFilterString(): string {
    return `drop-shadow(${toSVGNumber( this.offset.x )}px ${toSVGNumber( this.offset.y )}px ${toSVGNumber( this.blurRadius )}px ${this.colorCSS})`;
  }

  isDOMCompatible() {
    return true;
  }

  applyCanvasFilter( wrapper: CanvasContextWrapper ): void {
    throw new Error( 'unimplemented' );
  }

  applySVGFilter( svgFilter: SVGFilterElement, inName: string, resultName?: string ): void {
    throw new Error( 'unimplemented' );
  }
}

scenery.register( 'DropShadow', DropShadow );
export default DropShadow;