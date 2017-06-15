// Copyright 2013-2017, University of Colorado Boulder

/**
 * Abstract base type for LinearGradient and RadialGradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var Color = require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Paint = require( 'SCENERY/util/Paint' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  // TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
  function Gradient() {
    assert && assert( this.constructor.name !== 'Gradient',
      'Please create a LinearGradient or RadialGradient. Do not directly use the supertype Gradient.' );

    Paint.call( this );

    // @private {Array.<{ ratio: {number}, stop: {...} }>}
    this.stops = [];

    // @private {number}
    this.lastStopRatio = 0;

    // @private {CanvasGradient|null} - Lazily created
    this.canvasGradient = null;

    // @private {boolean} - Whether we should force a check of whether stops have changed
    this.colorStopsDirty = false;

    // @private {Array.<string>} - Used to check to see if colors have changed since last time
    this.lastColorStopValues = [];
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

      this.lastColorStopValues.push( '' ); // So it's the same length

      return this;
    },

    /**
     * Subtypes should return a fresh CanvasGradient type.
     * @protected
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
      this.colorStopsDirty = true;
    },

    // TODO doc @private
    haveCanvasColorStopsChanged: function() {
      if ( this.lastColorStopValues === null ) {
        return true;
      }

      for ( var i = 0; i < this.stops.length; i++ ) {
        if ( Gradient.colorToString( this.stops[ i ].color ) !== this.lastColorStopValues[ i ] ) {
          return true;
        }
      }

      return false;
    },

    getCanvasStyle: function() {
      // Check if we need to regenerate the Canvas gradient
      if ( !this.canvasGradient || ( this.colorStopsDirty && this.haveCanvasColorStopsChanged() ) ) {
        this.colorStopsDirty = false;

        cleanArray( this.lastColorStopValues );
        this.canvasGradient = this.createCanvasGradient();

        for ( var i = 0; i < this.stops.length; i++ ) {
          var stop = this.stops[ i ];

          var colorString = Gradient.colorToString( stop.color );
          this.canvasGradient.addColorStop( stop.ratio, colorString );

          // Save it so we can compare next time whether our generated gradient would have changed
          this.lastColorStopValues.push( colorString );
        }
      }

      return this.canvasGradient;
    }
  } );

  /**
   * Returns the current value of the generally-allowed color types for Gradient, as a string.
   * @public
   *
   * @param {Color|string|Property.<Color|string|null>|null} color
   * @returns {string}
   */
  Gradient.colorToString = function( color ) {
    // to {Color|string|null}
    if ( color instanceof Property ) {
      color = color.value;
    }

    // to {Color|string}
    if ( color === null ) {
      color = 'transparent';
    }

    // to {string}
    if ( color instanceof Color ) {
      color = color.toCSS();
    }

    return color;
  };

  return Gradient;
} );
