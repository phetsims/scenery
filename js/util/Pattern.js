// Copyright 2013-2017, University of Colorado Boulder

/**
 * A pattern that will deliver a fill or stroke that will repeat an image in both directions (x and y).
 *
 * TODO: future support for repeat-x, repeat-y or no-repeat (needs SVG support)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Paint = require( 'SCENERY/util/Paint' );
  var scenery = require( 'SCENERY/scenery' );
  var SVGPattern = require( 'SCENERY/display/SVGPattern' );

  // TODO: support scene or other various content (SVG is flexible, can backport to canvas)
  // TODO: investigate options to support repeat-x, repeat-y or no-repeat in SVG (available repeat options from Canvas)
  function Pattern( image ) {
    Paint.call( this );

    this.image = image;

    // use the global scratch canvas instead of creating a new Canvas
    this.canvasPattern = scenery.scratchContext.createPattern( image, 'repeat' );
  }

  scenery.register( 'Pattern', Pattern );

  inherit( Paint, Pattern, {
    isPattern: true,

    getCanvasStyle: function() {
      return this.canvasPattern;
    },

    /**
     * Creates an SVG paint object for creating/updating the SVG equivalent definition.
     * @public
     *
     * @param {SVGBlock} svgBlock
     * @returns {SVGGradient|SVGPattern}
     */
    createSVGPaint: function( svgBlock ) {
      return SVGPattern.createFromPool( this );
    },

    toString: function() {
      return 'new scenery.Pattern( $( \'<img src="' + this.image.src + '"/>\' )[0] )';
    }
  } );

  return Pattern;
} );
