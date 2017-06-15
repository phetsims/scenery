// Copyright 2013-2015, University of Colorado Boulder

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
   * @param {LinearGradient} gradient
   */
  function SVGLinearGradient( gradient ) {
    this.initialize( gradient );
  }

  scenery.register( 'SVGLinearGradient', SVGLinearGradient );

  inherit( Object, SVGLinearGradient, {
    /**
     * Poolable initializer.
     * @private
     *
     * @param {LinearGradient} linearGradient
     */
    initialize: function( linearGradient ) {
      SVGGradient.prototype.initialize.call( this, linearGradient );

      // Linear-specific setup
      this.definition.setAttribute( 'x1', linearGradient.start.x );
      this.definition.setAttribute( 'y1', linearGradient.start.y );
      this.definition.setAttribute( 'x2', linearGradient.end.x );
      this.definition.setAttribute( 'y2', linearGradient.end.y );
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

  Poolable.mixin( SVGLinearGradient, {
    constructorDuplicateFactory: function( pool ) {
      return function( gradient ) {
        if ( pool.length ) {
          return pool.pop().initialize( gradient );
        }
        else {
          return new SVGLinearGradient( gradient );
        }
      };
    }
  } );

  return SVGLinearGradient;
} );
