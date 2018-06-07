// Copyright 2018, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Paint = require( 'SCENERY/util/Paint' );
  var PaintObserver = require( 'SCENERY/display/PaintObserver' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   * @extends {Property.<Color>}
   *
   * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
   * @param {Object} [options]
   */
  function PaintColorProperty( paint, options ) {
    var initialColor = Paint.toColor( paint );

    options = _.extend( {
      // {number} - 0 applies no change. Positive numbers brighten the color up to 1 (white). Negative numbers darken
      // the color up to -1 (black).
      brightnessAdjustment: 0
    }, options );

    Property.call( this, initialColor, options );

    // @private {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
    this._paint = null;

    // @private {number}
    this._brightnessAdjustment = options.brightnessAdjustment;

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
     * @param {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern} paint
     */
    setPaint: function( paint ) {
      assert && assert( Paint.isPaint( paint ) );

      this._paint = paint;
      this._paintObserver.setPrimary( paint );
    },
    set paint( value ) { this.setPaint( value ); },

    /**
     * Returns the current paint.
     * @public
     *
     * @returns {null|string|Color|Property.<string|Color>|LinearGradient|RadialGradient|Pattern}
     */
    getPaint: function() {
      return this._paint;
    },
    get paint() { return this.getPaint(); },

    /**
     * Sets the current brightness adjustment.
     * @public
     *
     * 0 applies no change. Positive numbers brighten the color up to 1 (white). Negative numbers darken the color up
     * to -1 (black).
     *
     * @param {number} brightnessAdjustment
     */
    setBrightnessAdjustment: function( brightnessAdjustment ) {
      assert && assert( typeof brightnessAdjustment === 'number' && brightnessAdjustment >= -1 && brightnessAdjustment <= 1 );

      if ( this.brightnessAdjustment !== brightnessAdjustment ) {
        this._brightnessAdjustment = brightnessAdjustment;

        this.invalidatePaint();
      }
    },
    set brightnessAdjustment( value ) { this.setBrightnessAdjustment( value ); },

    /**
     * Returns the current brightness adjustment.
     * @public
     *
     * @returns {number}
     */
    getBrightnessAdjustment: function() {
      return this._brightnessAdjustment;
    },
    get brightnessAdjustment() { return this.getBrightnessAdjustment(); },

    /**
     * Updates the value of this property.
     * @private
     */
    invalidatePaint: function() {
      this.value = Paint.toColor( this._paint ).colorUtilsBrightness( this._brightnessAdjustment );
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
