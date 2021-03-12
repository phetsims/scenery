// Copyright 2013-2020, University of Colorado Boulder

/**
 * A radial gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Vector2 from '../../../dot/js/Vector2.js';
import platform from '../../../phet-core/js/platform.js';
import SVGRadialGradient from '../display/SVGRadialGradient.js';
import scenery from '../scenery.js';
import Gradient from './Gradient.js';

class RadialGradient extends Gradient {
  /**
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
   *
   * TODO: support Vector2s as p0 and p1
   *
   * @param {number} x0 - X coordinate of the start point (ratio 0) in the local coordinate frame
   * @param {number} y0 - Y coordinate of the start point (ratio 0) in the local coordinate frame
   * @param {number} r0 - Radius of the start point (ratio 0) in the local coordinate frame
   * @param {number} x1 - X coordinate of the end point (ratio 1) in the local coordinate frame
   * @param {number} y1 - Y coordinate of the end point (ratio 1) in the local coordinate frame
   * @param {number} r1 - Radius of the end point (ratio 1) in the local coordinate frame
   */
  constructor( x0, y0, r0, x1, y1, r1 ) {
    super();

    // @public {Vector2}
    this.start = new Vector2( x0, y0 );
    this.end = new Vector2( x1, y1 );

    // If we are using Safari, we need to work around incorrect gradient handling for now,
    // see https://github.com/phetsims/sun/issues/526
    if ( platform.safari ) {
      const x = ( x0 + x1 ) / 2;
      const y = ( y0 + y1 ) / 2;
      this.start.x = x;
      this.start.y = y;
      this.end.x = x;
      this.end.y = y;
    }

    // @public {number}
    this.startRadius = r0;
    this.endRadius = r1;

    // @public {Vector2} - linear function from radius to point on the line from start to end
    this.focalPoint = this.start.plus( this.end.minus( this.start ).times( this.startRadius / ( this.startRadius - this.endRadius ) ) );

    // @public {boolean}
    this.startIsLarger = this.startRadius > this.endRadius;

    // @public {Vector2}
    this.largePoint = this.startIsLarger ? this.start : this.end;

    // @public {number}
    this.maxRadius = Math.max( this.startRadius, this.endRadius );
    this.minRadius = Math.min( this.startRadius, this.endRadius );

    // make sure that the focal point is in both circles. SVG doesn't support rendering outside of them
    if ( this.startRadius >= this.endRadius ) {
      assert && assert( this.focalPoint.minus( this.start ).magnitude <= this.startRadius );
    }
    else {
      assert && assert( this.focalPoint.minus( this.end ).magnitude <= this.endRadius );
    }
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
    return scenery.scratchContext.createRadialGradient( this.start.x, this.start.y, this.startRadius, this.end.x, this.end.y, this.endRadius );
  }

  /**
   * Creates an SVG paint object for creating/updating the SVG equivalent definition.
   * @public
   *
   * @param {SVGBlock} svgBlock
   * @returns {SVGGradient|SVGPattern}
   */
  createSVGPaint( svgBlock ) {
    return SVGRadialGradient.createFromPool( svgBlock, this );
  }

  /**
   * Returns stops suitable for direct SVG use.
   * @public
   * @override
   *
   * NOTE: SVG has certain stop requirements, so we need to remap/reverse in some cases.
   *
   * @returns {Array.<{ ratio: {number}, stop: {Color|string|Property.<Color|string|null>|null} }>}
   */
  getSVGStops() {
    const startIsLarger = this.startIsLarger;
    const maxRadius = this.maxRadius;
    const minRadius = this.minRadius;

    //TODO: replace with dot.Utils.linear
    // maps x linearly from [a0,b0] => [a1,b1]
    function linearMap( a0, b0, a1, b1, x ) {
      return a1 + ( x - a0 ) * ( b1 - a1 ) / ( b0 - a0 );
    }

    function mapStop( stop ) {
      // flip the stops if the start has a larger radius
      let ratio = startIsLarger ? 1 - stop.ratio : stop.ratio;

      // scale the stops properly if the smaller radius isn't 0
      if ( minRadius > 0 ) {
        // scales our ratio from [0,1] => [minRadius/maxRadius,0]
        ratio = linearMap( 0, 1, minRadius / maxRadius, 1, ratio );
      }

      return {
        ratio: ratio,
        color: stop.color
      };
    }

    const stops = this.stops.map( mapStop );

    // switch the direction we apply stops in, so that the ratios always are increasing.
    if ( startIsLarger ) {
      stops.reverse();
    }

    return stops;
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    let result = 'new scenery.RadialGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.startRadius + ', ' + this.end.x + ', ' + this.end.y + ', ' + this.endRadius + ' )';

    _.each( this.stops, stop => {
      result += '.addColorStop( ' + stop.ratio + ', \'' + stop.color.toString() + '\' )';
    } );

    return result;
  }
}

RadialGradient.prototype.isRadialGradient = true;

scenery.register( 'RadialGradient', RadialGradient );
export default RadialGradient;