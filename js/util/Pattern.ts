// Copyright 2013-2023, University of Colorado Boulder

/**
 * A pattern that will deliver a fill or stroke that will repeat an image in both directions (x and y).
 *
 * TODO: future support for repeat-x, repeat-y or no-repeat (needs SVG support)
 * TODO: support scene or other various content (SVG is flexible, can backport to canvas)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { imageBitmapToCanvas, Paint, registerImageBitmap, scenery, SVGBlock, SVGPattern } from '../imports.js';

export type PatternImage = HTMLImageElement | HTMLCanvasElement | ImageBitmap;

export default class Pattern extends Paint {

  public image: PatternImage;
  public canvasPattern: CanvasPattern;

  /**
   * @param image - The image to use as a repeated pattern.
   */
  public constructor( image: PatternImage ) {
    super();

    this.image = image;

    if ( image instanceof HTMLImageElement ) {
      // Load it as an image bitmap, so that it can be used in Vello
      if ( image.width && image.height ) {
        registerImageBitmap( image );
      }
      else {
        // If it's not loaded yet, add a listener to register it when it is
        image.addEventListener( 'load', () => {
          if ( image.width && image.height ) {
            registerImageBitmap( image );
          }
        } );
      }
    }

    // Use the global scratch canvas instead of creating a new Canvas
    // @ts-expect-error TODO: scenery namespace
    this.canvasPattern = scenery.scratchContext.createPattern( image, 'repeat' );
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

  public getImageURL(): string {
    if ( this.image instanceof ImageBitmap ) {
      return imageBitmapToCanvas( this.image ).toDataURL();
    }
    else if ( this.image instanceof HTMLCanvasElement ) {
      return this.image.toDataURL();
    }
    else {
      return this.image.src;
    }
  }

  /**
   * Returns a string form of this object
   */
  public override toString(): string {
    return `new phet.scenery.Pattern( $( '<img src="${this.getImageURL()}"/>' )[0] )`;
  }

  public isPattern!: boolean;
}

Pattern.prototype.isPattern = true;

scenery.register( 'Pattern', Pattern );
