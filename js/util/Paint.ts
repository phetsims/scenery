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
  id: string;

  // (scenery-internal)
  transformMatrix: Matrix3 | null;

  constructor() {
    this.id = `paint${globalId++}`;
    this.transformMatrix = null;
  }

  /**
   * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
   */
  abstract getCanvasStyle(): string | CanvasGradient | CanvasPattern;

  /**
   * Sets how this paint (pattern/gradient) is transformed, compared with the local coordinate frame of where it is
   *
   * NOTE: This should only be used before the pattern/gradient is ever displayed.
   * TODO: Catch if this is violated?
   */
  setTransformMatrix( transformMatrix: Matrix3 ): this {
    if ( this.transformMatrix !== transformMatrix ) {
      this.transformMatrix = transformMatrix;
    }
    return this;
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   */
  abstract createSVGPaint( svgBlock: SVGBlock ): SVGGradient | SVGPattern;

  /**
   * Returns a string form of this object
   */
  toString(): string {
    return this.id;
  }

  isPaint!: boolean;
}

// TODO: can we remove this in favor of type checks?
Paint.prototype.isPaint = true;

scenery.register( 'Paint', Paint );
