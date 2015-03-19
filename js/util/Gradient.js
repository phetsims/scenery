// Copyright 2002-2014, University of Colorado Boulder


/**
 * Gradient base type for LinearGradient and RadialGradient. Will not function on its own
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Paint = require( 'SCENERY/util/Paint' );
  var Color = require( 'SCENERY/util/Color' );

  // TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
  scenery.Gradient = function Gradient( canvasGradient ) {
    assert && assert( this.constructor.name !== 'Gradient', 'Please create a LinearGradient or RadialGradient. Do not directly use the supertype Gradient.' );
    Paint.call( this );

    this.stops = [];
    this.lastStopRatio = 0;

    this.canvasGradient = canvasGradient;
  };
  var Gradient = scenery.Gradient;

  inherit( Paint, Gradient, {
    isGradient: true,

    /**
     * @param {Number} ratio        Monotonically increasing value in the range of 0 to 1
     * @param {Color|String} color  Color for the stop, either a scenery.Color or CSS color string
     */
    addColorStop: function( ratio, color ) {
      assert && assert( typeof ratio === 'number', 'Ratio needs to be a number' );
      assert && assert( ratio >= 0 && ratio <= 1, 'Ratio needs to be between 0,1 inclusively' );
      assert && assert( typeof color === 'string' || color instanceof Color,
        'Color should be a string or a {Color} object' );

      // TODO: invalidate the gradient?
      if ( this.lastStopRatio > ratio ) {
        // fail out, since browser quirks go crazy for this case
        throw new Error( 'Color stops not specified in the order of increasing ratios' );
      }
      else {
        this.lastStopRatio = ratio;
      }

      // make sure we have a scenery.Color now
      if ( typeof color === 'string' ) {
        color = new scenery.Color( color );
      }

      this.stops.push( {
        ratio: ratio,
        color: color
      } );

      // construct the Canvas gradient as we go
      this.canvasGradient.addColorStop( ratio, color.toCSS() );
      return this;
    },

    getCanvasStyle: function() {
      return this.canvasGradient;
    }
  } );

  return Gradient;
} );
