// Copyright 2018, University of Colorado Boulder

/**
 * A Property that will always hold a `Color` object representing the current value of a given paint (and can be set to
 * different paints).
 *
 * This is valuable, since:
 * ```
 *   const color = new scenery.Color( 'red' );
 *   const fill = new axon.Property( color );
 *   const paintColorProperty = new scenery.PaintColorProperty( fill );
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

define( require => {
  'use strict';

  const PaintDef = require( 'SCENERY/util/PaintDef' );
  const PaintObserver = require( 'SCENERY/display/PaintObserver' );
  const Property = require( 'AXON/Property' );
  const scenery = require( 'SCENERY/scenery' );

  class PaintColorProperty extends Property {

    /**
     * @extends {Property.<Color>}
     *
     * @param {PaintDef} paint
     * @param {Object} [options]
     */
    constructor( paint, options ) {
      const initialColor = PaintDef.toColor( paint );

      options = _.extend( {
        // {number} - 0 applies no change. Positive numbers brighten the color up to 1 (white). Negative numbers darken
        // the color up to -1 (black). See setLuminanceFactor() for more information.
        luminanceFactor: 0,

        // Property options
        useDeepEquality: true // We don't need to renotify for equivalent colors
      }, options );

      super( initialColor, options );

      // @private {PaintDef}
      this._paint = null;

      // @private {number} - See setLuminanceFactor() for more information.
      this._luminanceFactor = options.luminanceFactor;

      // @private {function} - Our "paint changed" listener, will update the value of this Property.
      this._changeListener = this.invalidatePaint.bind( this );

      // @private {PaintObserver}
      this._paintObserver = new PaintObserver( this._changeListener );

      this.setPaint( paint );
    }

    /**
     * Sets the current paint of the PaintColorProperty.
     * @public
     *
     * @param {PaintDef} paint
     */
    setPaint( paint ) {
      assert && assert( PaintDef.isPaintDef( paint ) );

      this._paint = paint;
      this._paintObserver.setPrimary( paint );
    }

    set paint( value ) { this.setPaint( value ); }

    /**
     * Returns the current paint.
     * @public
     *
     * @returns {PaintDef}
     */
    getPaint() {
      return this._paint;
    }

    get paint() { return this.getPaint(); }

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
     * @param {number} luminanceFactor
     */
    setLuminanceFactor( luminanceFactor ) {
      assert && assert( typeof luminanceFactor === 'number' && luminanceFactor >= -1 && luminanceFactor <= 1 );

      if ( this.luminanceFactor !== luminanceFactor ) {
        this._luminanceFactor = luminanceFactor;

        this.invalidatePaint();
      }
    }

    set luminanceFactor( value ) { this.setLuminanceFactor( value ); }

    /**
     * Returns the current value used for adjusting the brightness or darkness (luminance) of the color.
     * @public
     *
     * See setLuminanceFactor() for more information.
     *
     * @returns {number}
     */
    getLuminanceFactor() {
      return this._luminanceFactor;
    }

    get luminanceFactor() { return this.getLuminanceFactor(); }

    /**
     * Updates the value of this Property.
     * @private
     */
    invalidatePaint() {
      this.value = PaintDef.toColor( this._paint ).colorUtilsBrightness( this._luminanceFactor );
    }

    /**
     * Releases references.
     * @public
     * @override
     */
    dispose() {
      this.paint = null;

      super.dispose();
    }
  }

  return scenery.register( 'PaintColorProperty', PaintColorProperty );
} );
