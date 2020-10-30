// Copyright 2020, University of Colorado Boulder

/**
 * DropShadow filter
 *
 * TODO: preventFit OR handle bounds increase (or both)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import scenery from '../scenery.js';
import ColorDef from './ColorDef.js';
import Filter from './Filter.js';
import PaintDef from './PaintDef.js';

class DropShadow extends Filter {
  /**
   * @param {Vector2} offset
   * @param {number} blurRadius
   * @param {ColorDef} color
   * @param {number} [filterRegionPercentage]
   */
  constructor( offset, blurRadius, color, filterRegionPercentage = 15 ) {
    assert && assert( offset instanceof Vector2, 'DropShadow offset should be a Vector2' );
    assert && assert( offset.isFinite(), 'DropShadow offset should be finite' );
    assert && assert( typeof blurRadius === 'number', 'DropShadow blurRadius should be a number' );
    assert && assert( isFinite( blurRadius ), 'DropShadow blurRadius should be finite' );
    assert && assert( blurRadius >= 0, 'DropShadow blurRadius should be non-negative' );
    assert && assert( ColorDef.isColorDef( color ), 'DropShadow color should be a ColorDef' );

    super();

    // TODO: consider linking to the ColorDef (if it's a Property), or indicating that we need an update

    // @public {Vector2}
    this.offset = offset;

    // @public {number}
    this.blurRadius = blurRadius;

    // @public {ColorDef}
    this.color = color;

    // @private {string}
    this.colorCSS = PaintDef.toColor( color ).toCSS();

    this.filterRegionPercentageIncrease = filterRegionPercentage;
  }

  /**
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `drop-shadow(${toSVGNumber( this.offset.x )}px ${toSVGNumber( this.offset.y )}px ${toSVGNumber( this.blurRadius )}px ${this.colorCSS})`;
  }

  /**
   * @public
   * @override
   *
   * @returns {*}
   */
  isDOMCompatible() {
    return true;
  }
}

scenery.register( 'DropShadow', DropShadow );
export default DropShadow;