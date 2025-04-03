// Copyright 2013-2025, University of Colorado Boulder

/**
 * A pattern that will deliver a fill or stroke that will repeat an image in both directions (x and y).
 *
 * TODO: future support for repeat-x, repeat-y or no-repeat (needs SVG support) https://github.com/phetsims/scenery/issues/1581
 * TODO: support scene or other various content (SVG is flexible, can backport to canvas)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Paint from '../util/Paint.js';
import scenery from '../scenery.js';
import SVGBlock from '../display/SVGBlock.js';
import SVGPattern from '../display/SVGPattern.js';
import { scratchContext } from './scratches.js';

export default class Pattern extends Paint {

  public image: HTMLImageElement;
  public canvasPattern: CanvasPattern;

  /**
   * @param image - The image to use as a repeated pattern.
   */
  public constructor( image: HTMLImageElement ) {
    super();

    this.image = image;

    // Use the global scratch canvas instead of creating a new Canvas
    // @ts-expect-error TODO: scenery namespace https://github.com/phetsims/scenery/issues/1581
    this.canvasPattern = scratchContext.createPattern( image, 'repeat' );
  }


  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   */
  public getCanvasStyle(): CanvasPattern {
    return this.canvasPattern;
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   */
  public createSVGPaint( svgBlock: SVGBlock ): SVGPattern {
    return SVGPattern.pool.create( this );
  }

  /**
   * Returns a string form of this object
   */
  public override toString(): string {
    return `Pattern( ${this.image.src} )`;
  }

  public isPattern!: boolean;
}

Pattern.prototype.isPattern = true;

scenery.register( 'Pattern', Pattern );