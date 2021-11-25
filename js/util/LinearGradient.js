// Copyright 2013-2021, University of Colorado Boulder

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import { scenery, Gradient, SVGLinearGradient } from '../imports.js';

class LinearGradient extends Gradient {
  /**
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
   *
   * @param {number} x0 - X coordinate of the start point (ratio 0) in the local coordinate frame
   * @param {number} y0 - Y coordinate of the start point (ratio 0) in the local coordinate frame
   * @param {number} x1 - X coordinate of the end point (ratio 1) in the local coordinate frame
   * @param {number} y1 - Y coordinate of the end point (ratio 1) in the local coordinate frame
   */
  constructor( x0, y0, x1, y1 ) {
    assert && assert( isFinite( x0 ) && isFinite( y0 ) && isFinite( x1 ) && isFinite( y1 ) );

    // TODO: are we using this alternative format?
    const usesVectors = y1 === undefined;
    if ( usesVectors ) {
      assert && assert( ( x0 instanceof Vector2 ) && ( y0 instanceof Vector2 ), 'If less than 4 parameters are given, the first two parameters must be Vector2' );
    }

    super();

    this.start = usesVectors ? x0 : new Vector2( x0, y0 );
    this.end = usesVectors ? y0 : new Vector2( x1, y1 );
  }


  /**
   * Returns a fresh gradient given the starting parameters
   * @protected
   * @override
   *
   * @returns {CanvasGradient}
   */
  createCanvasGradient() {
    // use the global scratch canvas instead of creating a new Canvas
    return scenery.scratchContext.createLinearGradient( this.start.x, this.start.y, this.end.x, this.end.y );
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   * @public
   *
   * @param {SVGBlock} svgBlock
   * @returns {SVGGradient|SVGPattern}
   */
  createSVGPaint( svgBlock ) {
    return SVGLinearGradient.createFromPool( svgBlock, this );
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    let result = `new scenery.LinearGradient( ${this.start.x}, ${this.start.y}, ${this.end.x}, ${this.end.y} )`;

    _.each( this.stops, stop => {
      result += `.addColorStop( ${stop.ratio}, '${stop.color.toCSS ? stop.color.toCSS() : stop.color.toString()}' )`;
    } );

    return result;
  }
}


LinearGradient.prototype.isLinearGradient = true;

scenery.register( 'LinearGradient', LinearGradient );
export default LinearGradient;