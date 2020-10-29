// Copyright 2013-2020, University of Colorado Boulder

/**
 * A pattern that will deliver a fill or stroke that will repeat an image in both directions (x and y).
 *
 * TODO: future support for repeat-x, repeat-y or no-repeat (needs SVG support)
 * TODO: support scene or other various content (SVG is flexible, can backport to canvas)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import SVGPattern from '../display/SVGPattern.js';
import scenery from '../scenery.js';
import Paint from './Paint.js';

class Pattern extends Paint {
  /**
   * @param {HTMLImageElement} image - The image to use as a repeated pattern.
   */
  constructor( image ) {
    super();

    // @public {HTMLImageElement}
    this.image = image;

    // @public {CanvasPattern} - use the global scratch canvas instead of creating a new Canvas
    this.canvasPattern = scenery.scratchContext.createPattern( image, 'repeat' );
  }


  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   * @public
   * @override
   *
   * @returns {*}
   */
  getCanvasStyle() {
    return this.canvasPattern;
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   * @public
   *
   * @param {SVGBlock} svgBlock
   * @returns {SVGGradient|SVGPattern}
   */
  createSVGPaint( svgBlock ) {
    return SVGPattern.createFromPool( this );
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return 'new scenery.Pattern( $( \'<img src="' + this.image.src + '"/>\' )[0] )';
  }
}

// @public {boolean}
Pattern.prototype.isPattern = true;

scenery.register( 'Pattern', Pattern );
export default Pattern;