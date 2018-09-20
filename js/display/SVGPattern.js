// Copyright 2017, University of Colorado Boulder

/**
 * Creates an SVG pattern element for a given pattern.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @constructor
   *
   * @param {Pattern} pattern
   */
  function SVGPattern( pattern ) {
    this.initialize( pattern );
  }

  scenery.register( 'SVGPattern', SVGPattern );

  inherit( Object, SVGPattern, {
    /**
     * Poolable initializer.
     * @private
     *
     * @param {Pattern} pattern
     */
    initialize: function( pattern ) {
      sceneryLog && sceneryLog.Paints && sceneryLog.Paints( '[SVGPattern] initialize: ' + pattern.id );
      sceneryLog && sceneryLog.Paints && sceneryLog.push();

      var hasPreviousDefinition = this.definition !== undefined;

      // @public {SVGPatternElement} - persistent
      this.definition = this.definition || document.createElementNS( scenery.svgns, 'pattern' );

      if ( !hasPreviousDefinition ) {
        // so we don't depend on the bounds of the object being drawn with the pattern
        this.definition.setAttribute( 'patternUnits', 'userSpaceOnUse' );

        //TODO: is this needed?
        this.definition.setAttribute( 'patternContentUnits', 'userSpaceOnUse' );
      }

      if ( pattern.transformMatrix ) {
        this.definition.setAttribute( 'patternTransform', pattern.transformMatrix.getSVGTransform() );
      }
      else {
        this.definition.removeAttribute( 'patternTransform' );
      }

      this.definition.setAttribute( 'x', 0 );
      this.definition.setAttribute( 'y', 0 );
      this.definition.setAttribute( 'width', pattern.image.width );
      this.definition.setAttribute( 'height', pattern.image.height );

      // @private {SVGImageElement} - persistent
      this.imageElement = this.imageElement || document.createElementNS( scenery.svgns, 'image' );
      this.imageElement.setAttribute( 'x', 0 );
      this.imageElement.setAttribute( 'y', 0 );
      this.imageElement.setAttribute( 'width', pattern.image.width + 'px' );
      this.imageElement.setAttribute( 'height', pattern.image.height + 'px' );
      this.imageElement.setAttributeNS( scenery.xlinkns, 'xlink:href', pattern.image.src );
      if ( !hasPreviousDefinition ) {
        this.definition.appendChild( this.imageElement );
      }

      sceneryLog && sceneryLog.Paints && sceneryLog.pop();

      return this;
    },

    /**
     * Called from SVGBlock, matches other paints.
     * @public
     */
    update: function() {
      // Nothing
    },

    /**
     * Disposes, so that it can be reused from the pool.
     * @public
     */
    dispose: function() {
      this.freeToPool();
    }
  } );

  Poolable.mixInto( SVGPattern, {
    initialize: SVGPattern.prototype.initialize
  } );

  return SVGPattern;
} );
