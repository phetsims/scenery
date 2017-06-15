// Copyright 2013-2017, University of Colorado Boulder

/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Gradient = require( 'SCENERY/util/Gradient' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGLinearGradient = require( 'SCENERY/display/SVGLinearGradient' );
  var Vector2 = require( 'DOT/Vector2' );

  /**
   * @constructor
   * @extends Gradient
   *
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
   *
   * @param {number} x0 - X coordinate of the start point (ratio 0) in the local coordinate frame
   * @param {number} y0 - Y coordinate of the start point (ratio 0) in the local coordinate frame
   * @param {number} x1 - X coordinate of the end point (ratio 1) in the local coordinate frame
   * @param {number} y1 - Y coordinate of the end point (ratio 1) in the local coordinate frame
   */
  function LinearGradient( x0, y0, x1, y1 ) {
    assert && assert( isFinite( x0 ) && isFinite( y0 ) && isFinite( x1 ) && isFinite( y1 ) );

    // TODO: are we using this alternative format?
    var usesVectors = y1 === undefined;
    if ( usesVectors ) {
      assert && assert( ( x0 instanceof Vector2 ) && ( y0 instanceof Vector2 ), 'If less than 4 parameters are given, the first two parameters must be Vector2' );
    }

    this.start = usesVectors ? x0 : new Vector2( x0, y0 );
    this.end = usesVectors ? y0 : new Vector2( x1, y1 );

    Gradient.call( this );
  }

  scenery.register( 'LinearGradient', LinearGradient );

  inherit( Gradient, LinearGradient, {

    isLinearGradient: true,

    /**
     * Returns a fresh gradient given the starting parameters
     * @protected
     * @override
     *
     * @returns {CanvasGradient}
     */
    createCanvasGradient: function() {
      // use the global scratch canvas instead of creating a new Canvas
      return scenery.scratchContext.createLinearGradient( this.start.x, this.start.y, this.end.x, this.end.y );
    },

    /**
     * Creates an SVG paint object for creating/updating the SVG equivalent definition.
     * @public
     *
     * @param {SVGBlock} svgBlock
     * @returns {SVGGradient|SVGPattern}
     */
    createSVGPaint: function( svgBlock ) {
      return SVGLinearGradient.createFromPool( svgBlock, this );
    },

    toString: function() {
      var result = 'new scenery.LinearGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.end.x + ', ' + this.end.y + ' )';

      _.each( this.stops, function( stop ) {
        result += '.addColorStop( ' + stop.ratio + ', \'' + ( stop.color.toCSS ? stop.color.toCSS() : stop.color.toString() ) + '\' )';
      } );

      return result;
    }
  } );

  return LinearGradient;
} );
