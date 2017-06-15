// Copyright 2013-2016, University of Colorado Boulder

/**
 * Abstract base type for LinearGradient and RadialGradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Color = require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Paint = require( 'SCENERY/util/Paint' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  // TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
  function Gradient( canvasGradientFactory ) {
    assert && assert( this.constructor.name !== 'Gradient',
      'Please create a LinearGradient or RadialGradient. Do not directly use the supertype Gradient.' );

    Paint.call( this );

    // @private {Array.<{ ratio: {number}, stop: {...} }>}
    this.stops = [];

    // @private {number}
    this.lastStopRatio = 0;

    // @private {CanvasGradient|null} - Lazily created
    this.canvasGradient = null;

    // @private {CanvasGradient}
    this.canvasGradientFactory = canvasGradientFactory;
  }

  scenery.register( 'Gradient', Gradient );

  inherit( Paint, Gradient, {
    isGradient: true,

    /**
     * @param {number} ratio        Monotonically increasing value in the range of 0 to 1
     * @param {Color|String|Property.<Color|string>|null} color  Color for the stop, either a scenery.Color or CSS color string
     */
    addColorStop: function( ratio, color ) {
      assert && assert( typeof ratio === 'number', 'Ratio needs to be a number' );
      assert && assert( ratio >= 0 && ratio <= 1, 'Ratio needs to be between 0,1 inclusively' );
      assert && assert( color === null ||
                        typeof color === 'string' ||
                        color instanceof Color ||
                        ( color instanceof Property && ( typeof color.value === 'string' ||
                                                         color.value instanceof Color ) ),
        'Color should be a string or a {Color} object' );

      // TODO: invalidate the gradient?
      if ( this.lastStopRatio > ratio ) {
        // fail out, since browser quirks go crazy for this case
        throw new Error( 'Color stops not specified in the order of increasing ratios' );
      }
      else {
        this.lastStopRatio = ratio;
      }

      this.stops.push( {
        ratio: ratio,
        color: color
      } );

      // construct the Canvas gradient as we go
      this.canvasGradient.addColorStop( ratio, ( typeof color === 'string' ) ? color : color.toCSS() );
      return this;
    },

    /**
     * Subtypes should return a fresh CanvasGradient type.
     * @public
     * @abstract
     *
     * @returns {CanvasGradient}
     */
    createCanvasGradient: function() {
      throw new Error( 'abstract method' );
    },

    /**
     * Returns stops suitable for direct SVG use.
     * @public
     *
     * @returns {Array.<{ ratio: {number}, stop: {Color|string|Property.<Color|string|null>|null} }>}
     */
    getSVGStops: function() {
      return this.stops;
    },

    invalidateCanvasGradient: function() {
      this.canvasGradient = null;
    },

    getCanvasStyle: function() {
      if ( !this.canvasGradient ) {
        this.canvasGradient = this.canvasGradientFactory();

      }
      return this.canvasGradient;
    }
  } );

  return Gradient;
} );
