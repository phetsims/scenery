// Copyright 2017, University of Colorado Boulder

/**
 * Handles creation of an SVG stop element, and handles keeping it updated based on property/color changes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Color = require( 'SCENERY/util/Color' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  var scratchColor = new Color( 'transparent' );

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param {SVGGradient} svgGradient
   * @param {number} ratio
   * @param {Color|string|Property.<Color|string|null>|null} color
   */
  function SVGGradientStop( svgGradient, ratio, color ) {
    this.initialize( svgGradient, ratio, color );
  }

  scenery.register( 'SVGGradientStop', SVGGradientStop );

  inherit( Object, SVGGradientStop, {
    /**
     * Poolable initializer.
     * @private
     *
     * @param {SVGGradient} svgGradient
     * @param {number} ratio
     * @param {Color|string|Property.<Color|string|null>|null} color
     */
    initialize: function( svgGradient, ratio, color ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] initialize: ' + svgGradient.gradient.id + ' : ' + ratio );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      // @private {SVGGradient} - transient
      this.svgGradient = svgGradient;

      // @private {Color|string|Property.<Color|string|null>|null} - transient
      this.color = color;

      // @private {number}
      this.ratio = ratio;

      // @public {SVGStopElement} - persistent
      this.svgElement = this.svgElement || document.createElementNS( scenery.svgns, 'stop' );

      this.svgElement.setAttribute( 'offset', ratio );

      // @private {boolean}
      this.dirty = true; // true here so our update() actually properly initializes

      this.update();

      // @private {function} - persistent
      this.propertyListener = this.propertyListener || this.onPropertyChange.bind( this );
      this.colorListener = this.colorListener || this.markDirty.bind( this );

      if ( color instanceof Property ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] adding Property listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
        color.lazyLink( this.propertyListener );
        if ( color.value instanceof Color ) {
          sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] adding Color listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
          color.value.changeEmitter.addListener( this.colorListener );
        }
      }
      else if ( color instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] adding Color listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
        color.changeEmitter.addListener( this.colorListener );
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();

      return this;
    },

    /**
     * Called when our color is a Property and it changes.
     * @private
     *
     * @param {Color|string|null} newValue
     * @param {Color|string|null} oldValue
     */
    onPropertyChange: function( newValue, oldValue ) {
      if ( oldValue instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] removing Color listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
        oldValue.changeEmitter.removeListener( this.colorListener );
      }
      if ( newValue instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] adding Color listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
        newValue.changeEmitter.addListener( this.colorListener );
      }

      this.markDirty();
    },

    /**
     * Should be called when the color stop's value may have changed.
     * @private
     */
    markDirty: function() {
      this.dirty = true;
      this.svgGradient.markDirty();
    },

    /**
     * Updates the color stop to whatever the current color should be.
     * @public
     */
    update: function() {
      if ( !this.dirty ) {
        return;
      }
      this.dirty = false;

      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] update: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      // {Color|string|Property.<Color|string|null>|null}
      var color = this.color;

      // to {Color|string|null}
      if ( color instanceof Property ) {
        color = color.value;
      }

      // to {Color|string}
      if ( color === null ) {
        color = 'transparent';
      }

      // to {Color}, in our scratchColor
      if ( typeof color === 'string' ) {
        scratchColor.setCSS( color );
      }
      else {
        scratchColor.set( color );
      }

      // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
      // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
      // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
      // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
      var stopOpacityRule = 'stop-opacity: ' + scratchColor.a.toFixed( 20 ) + ';';

      // For GC, mutate the color so it is just RGB and output that CSS also
      scratchColor.alpha = 1;
      var stopColorRule = 'stop-color: ' + scratchColor.toCSS() + ';';

      this.svgElement.setAttribute( 'style', stopColorRule + ' ' + stopOpacityRule );

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Disposes, so that it can be reused from the pool.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] dispose: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      var color = this.color;

      if ( color instanceof Property ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] removing Property listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
        if ( color.hasListener( this.propertyListener ) ) {
          color.unlink( this.propertyListener );
        }
        if ( color.value instanceof Color ) {
          sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] removing Color listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
          color.value.changeEmitter.removeListener( this.colorListener );
        }
      }
      else if ( color instanceof Color ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradientStop] removing Color listener: ' + this.svgGradient.gradient.id + ' : ' + this.ratio );
        color.changeEmitter.removeListener( this.colorListener );
      }

      this.color = null; // clear the reference
      this.svgGradient = null; // clear the reference

      this.freeToPool();

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }
  } );

  Poolable.mixInto( SVGGradientStop, {
    initialize: SVGGradientStop.prototype.initialize
  } );

  return SVGGradientStop;
} );
