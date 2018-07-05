// Copyright 2018, University of Colorado Boulder

/**
 * A Property that will always hold a `Color` object representing the current value of a given paint (and can be set to
 * different paints).
 *
 * This is valuable, since:
 * ```
 *   var color = new scenery.Color( 'red' );
 *   var fill = new axon.Property( color );
 *   var paintColorProperty = new scenery.PaintColorProperty( fill );
 *
 *   // value is converted to a {Color}
 *   paintColorProperty.value; // r: 255, g: 0, b: 0, a: 1
 *
 *   // watches direct Color mutation
 *   color.red = 128;
 *   paintColorProperty.value; // r: 128, g: 0, b: 0, a: 1
 *
 *   // watches the Property mutation
 *   fill.value = 'green';
 *   paintColorProperty.value; // r: 0, g: 128, b: 0, a: 1
 *
 *   // can switch to a different paint
 *   paintColorProperty.paint = 'blue';
 *   paintColorProperty.value; // r: 0, g: 0, b: 255, a: 1
 * ```
 *
 * Basically, you don't have to add your own listeners to both (optionally) any Properties in a paint and (optionally)
 * any Color objects (since it's all handled).
 *
 * This is particularly helpful to create paints that are either lighter or darker than an original paint (where it
 * will update its color value when the original is updated).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var PaintDef = require( 'SCENERY/util/PaintDef' );
  var PaintObserver = require( 'SCENERY/display/PaintObserver' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   * @extends {Property.<Color>}
   *
   * @param {PaintDef} paint
   * @param {Object} [options]
   */
  function PaintColorProperty( paint, options ) {
    var initialColor = PaintDef.toColor( paint );

    options = _.extend( {
      // {number} - 0 applies no change. Positive numbers brighten the color up to 1 (white). Negative numbers darken
      // the color up to -1 (black). See setFactor() for more information.
      factor: 0,

      // Property options
      useDeepEquality: true // We don't need to renotify for equivalent colors
    }, options );

    Property.call( this, initialColor, options );

    // @private {PaintDef}
    this._paint = null;

    // @private {number} - See setFactor() for more information.
    this._factor = options.factor;

    // @private {function} - Our "paint changed" listener, will update the value of this Property.
    this._changeListener = this.invalidatePaint.bind( this );

    // @private {PaintObserver}
    this._paintObserver = new PaintObserver( this._changeListener );

    this.setPaint( paint );
  }

  scenery.register( 'PaintColorProperty', PaintColorProperty );

  inherit( Property, PaintColorProperty, {
    /**
     * Sets the current paint of the PaintColorProperty.
     * @public
     *
     * @param {PaintDef} paint
     */
    setPaint: function( paint ) {
      assert && assert( PaintDef.isPaintDef( paint ) );

      this._paint = paint;
      this._paintObserver.setPrimary( paint );
    },
    set paint( value ) { this.setPaint( value ); },

    /**
     * Returns the current paint.
     * @public
     *
     * @returns {PaintDef}
     */
    getPaint: function() {
      return this._paint;
    },
    get paint() { return this.getPaint(); },

    /**
     * Sets the current value used for adjusting the brightness or darkness (luminance) of the color.
     * @public
     *
     * If this factor is a non-zero value, the value of this Property will be either a brightened or darkened version of
     * the paint (depending on the value of the factor). 0 applies no change. Positive numbers brighten the color up to
     * 1 (white). Negative numbers darken the color up to -1 (black).
     *
     * For example, if the given paint is blue, the below factors will result in:
     *
     *   -1: black
     * -0.5: dark blue
     *    0: blue
     *  0.5: light blue
     *    1: white
     *
     * With intermediate values basically "interpolated". This uses the `Color` colorUtilsBrightness method to adjust
     * the paint.
     *
     * @param {number} factor
     */
    setFactor: function( factor ) {
      assert && assert( typeof factor === 'number' && factor >= -1 && factor <= 1 );

      if ( this.factor !== factor ) {
        this._factor = factor;

        this.invalidatePaint();
      }
    },
    set factor( value ) { this.setFactor( value ); },

    /**
     * Returns the current value used for adjusting the brightness or darkness (luminance) of the color.
     * @public
     *
     * See setFactor() for more information.
     *
     * @returns {number}
     */
    getFactor: function() {
      return this._factor;
    },
    get factor() { return this.getFactor(); },

    /**
     * Updates the value of this Property.
     * @private
     */
    invalidatePaint: function() {
      this.value = PaintDef.toColor( this._paint ).colorUtilsBrightness( this._factor );
    },

    /**
     * Releases references.
     * @public
     * @override
     */
    dispose: function() {
      this.paint = null;

      Property.prototype.dispose.call( this );
    }
  } );

  return PaintColorProperty;
} );
