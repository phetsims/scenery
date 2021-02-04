// Copyright 2020, University of Colorado Boulder

/**
 * HueRotate filter
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import toSVGNumber from '../../../dot/js/toSVGNumber.js';
import Utils from '../../../dot/js/Utils.js';
import scenery from '../scenery.js';
import ColorMatrixFilter from './ColorMatrixFilter.js';

class HueRotate extends ColorMatrixFilter {
  /**
   * @param {number} amount - In radians, the amount of hue to color-shift
   */
  constructor( amount ) {
    assert && assert( typeof amount === 'number', 'HueRotate amount should be a number' );
    assert && assert( isFinite( amount ), 'HueRotate amount should be finite' );
    assert && assert( amount >= 0, 'HueRotate amount should be non-negative' );

    const cos = Math.cos( amount );
    const sin = Math.sin( amount );

    // https://drafts.fxtf.org/filter-effects/#attr-valuedef-type-huerotate
    super(
      0.213 + 0.787 * cos - 0.213 * sin,
      0.715 - 0.715 * cos - 0.715 * sin,
      0.072 - 0.072 * cos + 0.928 * sin,
      0, 0,
      0.213 - 0.213 * cos + 0.143 * sin,
      0.715 + 0.285 * cos + 0.140 * sin,
      0.072 - 0.072 * cos - 0.283 * sin,
      0, 0,
      0.213 - 0.213 * cos - 0.787 * sin,
      0.715 - 0.715 * cos + 0.715 * sin,
      0.072 + 0.928 * cos + 0.072 * sin,
      0, 0,
      0, 0, 0, 1, 0
    );

    // @public {number}
    this.amount = amount;
  }

  /**
   * Returns the CSS-style filter substring specific to this single filter, e.g. `grayscale(1)`. This should be used for
   * both DOM elements (https://developer.mozilla.org/en-US/docs/Web/CSS/filter) and when supported, Canvas
   * (https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D/filter).
   * @public
   * @override
   *
   * @returns {string}
   */
  getCSSFilterString() {
    return `hue-rotate(${toSVGNumber( Utils.toDegrees( this.amount ) )}deg)`;
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

scenery.register( 'HueRotate', HueRotate );
export default HueRotate;