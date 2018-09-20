// Copyright 2017, University of Colorado Boulder

/**
 * Controller that creates and keeps an SVG radial gradient up-to-date with a Scenery RadialGradient
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
   * @param {RadialGradient} radialGradient
   */
  function SVGRadialGradient( svgBlock, radialGradient ) {
    this.initialize( svgBlock, radialGradient );
  }

  scenery.register( 'SVGRadialGradient', SVGRadialGradient );

  inherit( SVGGradient, SVGRadialGradient, {
    /**
     * Poolable initializer.
     * @private
     *
     * @param {SVGBlock} svgBlock
     * @param {RadialGradient} radialGradient
     */
    initialize: function( svgBlock, radialGradient ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGRadialGradient] initialize ' + radialGradient.id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      SVGGradient.prototype.initialize.call( this, svgBlock, radialGradient );

      // Radial-specific setup
      this.definition.setAttribute( 'cx', radialGradient.largePoint.x );
      this.definition.setAttribute( 'cy', radialGradient.largePoint.y );
      this.definition.setAttribute( 'r', radialGradient.maxRadius );
      this.definition.setAttribute( 'fx', radialGradient.focalPoint.x );
      this.definition.setAttribute( 'fy', radialGradient.focalPoint.y );

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();

      return this;
    },

    /**
     * Creates the gradient-type-specific definition.
     * @protected
     * @override
     *
     * @returns {SVGRadialGradientElement}
     */
    createDefinition: function() {
      return document.createElementNS( scenery.svgns, 'radialGradient' );
    }
  } );

  Poolable.mixInto( SVGRadialGradient, {
    initialize: SVGRadialGradient.prototype.initialize
  } );

  return SVGRadialGradient;
} );
