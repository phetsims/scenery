// Copyright 2017, University of Colorado Boulder

/**
 * Base type for controllers that create and keep an SVG gradient element up-to-date with a Scenery gradient.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGGradientStop = require( 'SCENERY/display/SVGGradientStop' );

  /**
   * @constructor
   *
   * @param {SVGBlock} svgBlock
   * @param {Gradient} gradient
   */
  function SVGGradient( svgBlock, gradient ) {
    this.initialize( svgBlock, gradient );
  }

  scenery.register( 'SVGGradient', SVGGradient );

  inherit( Object, SVGGradient, {
    /**
     * Poolable initializer.
     * @private
     *
     * @param {SVGBlock} svgBlock
     * @param {Gradient} gradient
     */
    initialize: function( svgBlock, gradient ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradient] initialize ' + gradient.id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      // @private {SVGBlock} - transient
      this.svgBlock = svgBlock;

      // @private {Gradient} - transient
      this.gradient = gradient;

      var hasPreviousDefinition = this.definition !== undefined;

      // @public {SVGGradientElement} - persistent
      this.definition = this.definition || this.createDefinition();

      if ( !hasPreviousDefinition ) {
        // so we don't depend on the bounds of the object being drawn with the gradient
        this.definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' );
      }

      if ( gradient.transformMatrix ) {
        this.definition.setAttribute( 'gradientTransform', gradient.transformMatrix.getSVGTransform() );
      }
      else {
        this.definition.removeAttribute( 'gradientTransform' );
      }

      // We need to make a function call, as stops need to be rescaled/reversed in some radial gradient cases.
      var gradientStops = gradient.getSVGStops();

      // @private {Array.<SVGGradientStop>} - transient
      this.stops = cleanArray( this.stops );
      for ( var i = 0; i < gradientStops.length; i++ ) {
        var stop = new SVGGradientStop( this, gradientStops[ i ].ratio, gradientStops[ i ].color );
        this.stops.push( stop );
        this.definition.appendChild( stop.svgElement );
      }

      // @private {boolean}
      this.dirty = false;

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Creates the gradient-type-specific definition.
     * @protected
     * @abstract
     *
     * @returns {SVGGradientElement}
     */
    createDefinition: function() {
      throw new Error( 'abstract method' );
    },

    /**
     * Called from SVGGradientStop when a stop needs to change the actual color.
     * @public
     */
    markDirty: function() {
      if ( !this.dirty ) {
        sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradient] switched to dirty: ' + this.gradient.id );
        sceneryLog && sceneryLog.Paints && sceneryLog.push();

        this.dirty = true;

        this.svgBlock.markDirtyGradient( this );

        sceneryLog && sceneryLog.Paints && sceneryLog.pop();
      }
    },

    /**
     * Called from SVGBlock when we need to update our color stops.
     * @public
     */
    update: function() {
      if ( !this.dirty ) {
        return;
      }
      this.dirty = false;

      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradient] update: ' + this.gradient.id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      for ( var i = 0; i < this.stops.length; i++ ) {
        this.stops[ i ].update();
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    },

    /**
     * Disposes, so that it can be reused from the pool.
     * @public
     */
    dispose: function() {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGGradient] dispose ' + this.gradient.id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      // Dispose and clean up stops
      for ( var i = 0; i < this.stops.length; i++ ) {
        var stop = this.stops[ i ]; // SVGGradientStop
        this.definition.removeChild( stop.svgElement );
        stop.dispose();
      }
      cleanArray( this.stops );

      this.svgBlock = null;
      this.gradient = null;

      this.freeToPool();

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();
    }
  } );

  return SVGGradient;
} );
