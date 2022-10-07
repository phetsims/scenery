// Copyright 2014-2022, University of Colorado Boulder

/**
 * Base type for gradients and patterns (and NOT the only type for fills/strokes)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import { scenery, SVGBlock, SVGGradient, SVGPattern } from '../imports.js';

let globalId = 1;

export default abstract class Paint {

  // (scenery-internal)
  public id: string;

  // (scenery-internal)
  public transformMatrix: Matrix3 | null;

  public constructor() {
    this.id = `paint${globalId++}`;
    this.transformMatrix = null;
  }

  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   */
  public abstract getCanvasStyle(): string | CanvasGradient | CanvasPattern;

  /**
   * Sets how this paint (pattern/gradient) is transformed, compared with the local coordinate frame of where it is
   *
   * NOTE: This should only be used before the pattern/gradient is ever displayed.
   * TODO: Catch if this is violated?
   *
   * NOTE: The scale should be symmetric if it will be used as a stroke. It is difficult to set a different x and y scale
   * for canvas at the same time.
   */
  public setTransformMatrix( transformMatrix: Matrix3 ): this {
    if ( this.transformMatrix !== transformMatrix ) {
      this.transformMatrix = transformMatrix;
    }
    return this;
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   */
  public abstract createSVGPaint( svgBlock: SVGBlock ): SVGGradient | SVGPattern;

  /**
   * Returns a string form of this object
   */
  public toString(): string {
    return this.id;
  }

  public isPaint!: boolean;
}

// TODO: can we remove this in favor of type checks?
Paint.prototype.isPaint = true;

scenery.register( 'Paint', Paint );
