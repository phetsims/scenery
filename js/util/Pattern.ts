// Copyright 2013-2022, University of Colorado Boulder

/**
 * A pattern that will deliver a fill or stroke that will repeat an image in both directions (x and y).
 *
 * TODO: future support for repeat-x, repeat-y or no-repeat (needs SVG support)
 * TODO: support scene or other various content (SVG is flexible, can backport to canvas)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery, Paint, SVGPattern, SVGBlock } from '../imports.js';

class Pattern extends Paint {

  image: HTMLImageElement;
  canvasPattern: CanvasPattern;

  /**
   * @param image - The image to use as a repeated pattern.
   */
  constructor( image: HTMLImageElement ) {
    super();

    this.image = image;

    // Use the global scratch canvas instead of creating a new Canvas
    // @ts-ignore TODO: scenery namespace
    this.canvasPattern = scenery.scratchContext.createPattern( image, 'repeat' );
  }


  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   */
  getCanvasStyle(): CanvasPattern {
    return this.canvasPattern;
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   */
  createSVGPaint( svgBlock: SVGBlock ): SVGPattern {
    return SVGPattern.pool.create( this );
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return `new scenery.Pattern( $( '<img src="${this.image.src}"/>' )[0] )`;
  }

  isPattern!: boolean;
}

Pattern.prototype.isPattern = true;

scenery.register( 'Pattern', Pattern );
export default Pattern;