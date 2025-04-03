// Copyright 2022-2025, University of Colorado Boulder

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import ColorDef from '../util/ColorDef.js';
import Gradient from '../util/Gradient.js';
import scenery from '../scenery.js';
import SVGBlock from '../display/SVGBlock.js';
import SVGLinearGradient from '../display/SVGLinearGradient.js';
import { scratchContext } from './scratches.js';

export default class LinearGradient extends Gradient {

  public start: Vector2;
  public end: Vector2;

  /**
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ] https://github.com/phetsims/scenery/issues/1581
   *
   * @param x0 - X coordinate of the start point (ratio 0) in the local coordinate frame
   * @param y0 - Y coordinate of the start point (ratio 0) in the local coordinate frame
   * @param x1 - X coordinate of the end point (ratio 1) in the local coordinate frame
   * @param y1 - Y coordinate of the end point (ratio 1) in the local coordinate frame
   */
  public constructor( x0: number, y0: number, x1: number, y1: number ) {
    assert && assert( isFinite( x0 ) && isFinite( y0 ) && isFinite( x1 ) && isFinite( y1 ) );

    super();

    this.start = new Vector2( x0, y0 );
    this.end = new Vector2( x1, y1 );
  }


  /**
   * Returns a fresh gradient given the starting parameters
   */
  public createCanvasGradient(): CanvasGradient {
    // use the global scratch canvas instead of creating a new Canvas
    return scratchContext.createLinearGradient( this.start.x, this.start.y, this.end.x, this.end.y );
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   */
  public createSVGPaint( svgBlock: SVGBlock ): SVGLinearGradient {
    return SVGLinearGradient.pool.create( svgBlock, this );
  }

  /**
   * Returns a string form of this object
   */
  public override toString(): string {
    let result = `LinearGradient( ${this.start.x}, ${this.start.y}, ${this.end.x}, ${this.end.y} )`;

    _.each( this.stops, stop => {
      result += `.addColorStop( ${stop.ratio}, ${ColorDef.scenerySerialize( stop.color )} )`;
    } );

    return result;
  }

  public isLinearGradient!: boolean;
}

LinearGradient.prototype.isLinearGradient = true;

scenery.register( 'LinearGradient', LinearGradient );