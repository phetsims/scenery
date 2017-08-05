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

  /**
   * @constructor
   * @extends Paint
   *
   * TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
   */
  function Gradient() {
    assert && assert( this.constructor.name !== 'Gradient',
      'Please create a LinearGradient or RadialGradient. Do not directly use the supertype Gradient.' );

    Paint.call( this );

    // @private {Array.<{ ratio: {number}, color: {...} }>}
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
    // @public {boolean}
    isGradient: true,

    /**
     * Adds a color stop to the gradient.
     * @public
     *
     * Color stops should be added in order (monotonically increasing ratio values).
     *
     * NOTE: Color stops should only be added before using the gradient as a fill/stroke. Adding stops afterwards
     *       will result in undefined behavior.
     * TODO: Catch attempts to do the above.
     *
     * @param {number} ratio - Monotonically increasing value in the range of 0 to 1
     * @param {Color|String|Property.<Color|string>|null} color
     * @returns {Gradient} - for chaining
     */
    addColorStop: function( ratio, color ) {
      assert && assert( typeof ratio === 'number', 'Ratio needs to be a number' );
      assert && assert( ratio >= 0 && ratio <= 1, 'Ratio needs to be between 0,1 inclusively' );
      assert && assert( color === null ||
                        typeof color === 'string' ||
                        color instanceof Color ||
                        ( color instanceof Property && ( color.value === null ||
                                                         typeof color.value === 'string' ||
                                                         color.value instanceof Color ) ),
        'Color should match the addColorStop type specification' );

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

      // Easiest to just push a value here, so that it is always the same length as the stops array.
      this.lastColorStopValues.push( '' );

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

    /**
     * Forces a re-check of whether colors have changed, so that the Canvas gradient can be regenerated if
     * necessary.
     * @public
     */
    invalidateCanvasGradient: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( 'Invalidated Canvas Gradient for #' + this.id );
      this.colorStopsDirty = true;
    },

    /**
     * Compares the current color values with the last-recorded values for the current Canvas gradient.
     * @private
     *
     * This is needed since the values of color properties (or the color itself) may change.
     *
     * @returns {boolean}
     */
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

    /**
     * Returns an object that can be passed to a Canvas context's fillStyle or strokeStyle.
     * @public
     * @override
     *
     * @returns {*}
     */
    getCanvasStyle: function() {
      // Check if we need to regenerate the Canvas gradient
      if ( !this.canvasGradient || ( this.colorStopsDirty && this.haveCanvasColorStopsChanged() ) ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( 'Regenerating Canvas Gradient for #' + this.id );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();

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

        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
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
