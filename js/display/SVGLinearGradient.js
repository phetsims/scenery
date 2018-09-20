// Copyright 2017, University of Colorado Boulder

/**
 * Controller that creates and keeps an SVG linear gradient up-to-date with a Scenery LinearGradient
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGGradient = require( 'SCENERY/display/SVGGradient' );

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param {SVGBlock} svgBlock
   * @param {LinearGradient} linearGradient
   */
  function SVGLinearGradient( svgBlock, linearGradient ) {
    this.initialize( svgBlock, linearGradient );
  }

  scenery.register( 'SVGLinearGradient', SVGLinearGradient );

  inherit( SVGGradient, SVGLinearGradient, {
    /**
     * Poolable initializer.
     * @private
     *
     * @param {SVGBlock} svgBlock
     * @param {LinearGradient} linearGradient
     */
    initialize: function( svgBlock, linearGradient ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGLinearGradient] initialize ' + linearGradient.id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      SVGGradient.prototype.initialize.call( this, svgBlock, linearGradient );

      // seems we need the defs: http://stackoverflow.com/questions/7614209/linear-gradients-in-svg-without-defs
      // SVG: spreadMethod 'pad' 'reflect' 'repeat' - find Canvas usage

      /* Approximate example of what we are creating:
       <linearGradient id="grad2" x1="0" y1="0" x2="100" y2="0" gradientUnits="userSpaceOnUse">
       <stop offset="0" style="stop-color:rgb(255,255,0);stop-opacity:1" />
       <stop offset="0.5" style="stop-color:rgba(255,255,0,0);stop-opacity:0" />
       <stop offset="1" style="stop-color:rgb(255,0,0);stop-opacity:1" />
       </linearGradient>
       */

      // Linear-specific setup
      this.definition.setAttribute( 'x1', linearGradient.start.x );
      this.definition.setAttribute( 'y1', linearGradient.start.y );
      this.definition.setAttribute( 'x2', linearGradient.end.x );
      this.definition.setAttribute( 'y2', linearGradient.end.y );

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();

      return this;
    },

    /**
     * Creates the gradient-type-specific definition.
     * @protected
     * @override
     *
     * @returns {SVGLinearGradientElement}
     */
    createDefinition: function() {
      return document.createElementNS( scenery.svgns, 'linearGradient' );
    }
  } );

  Poolable.mixInto( SVGLinearGradient, {
    initialize: SVGLinearGradient.prototype.initialize
  } );

  return SVGLinearGradient;
} );
