// Copyright 2022-2023, University of Colorado Boulder

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { ColorDef, Gradient, scenery, SVGBlock, SVGLinearGradient } from '../imports.js';

export default class LinearGradient extends Gradient {

  public start: Vector2;
  public end: Vector2;

  /**
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
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
    // @ts-expect-error TODO scenery namespace
    return scenery.scratchContext.createLinearGradient( this.start.x, this.start.y, this.end.x, this.end.y );
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
    let result = `new phet.scenery.LinearGradient( ${this.start.x}, ${this.start.y}, ${this.end.x}, ${this.end.y} )`;

    _.each( this.stops, stop => {
      result += `.addColorStop( ${stop.ratio}, ${ColorDef.scenerySerialize( stop.color )} )`;
    } );

    return result;
  }

  public isLinearGradient!: boolean;
}

LinearGradient.prototype.isLinearGradient = true;

scenery.register( 'LinearGradient', LinearGradient );
