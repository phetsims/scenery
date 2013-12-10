// Copyright 2002-2013, University of Colorado

/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying any changes at a later time.
 *
 * Node API needed:
 * {
 *   attachSVGDrawable: function( SVGSelfDrawable )
 *   detachSVGDrawable: function( SVGSelfDrawable )
 * }
 *
 * visual state API needed:
 * {
 *   markPaintDirty: function() // used by Strokable/Fillable
 *   node: Node                 // used by Strokable/Fillable
 *   drawable: SVGSelfDrawable  // set by the visual state on initialization     NOTE: required for any type of visual state! (used by Strokable/Fillable states)
 *   svgElement: SVGElement     // what is displayed in the SVG tree (should be the base of the displayed element)
 *   updateSVG: function()      // updates any visual state
 *   onDetach: function()       // called when the state is detached from a drawable. optionally discard DOM elements. we guarantee state will be
 *                              // initialized again before any more update() calls
 * }
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.SVGSelfDrawable = function SVGSelfDrawable( trail, renderer, instance ) {
    Drawable.call( this, trail, renderer );
    
    this.instance = instance;
  };
  var SVGSelfDrawable = scenery.SVGSelfDrawable;
  
  inherit( Drawable, SVGSelfDrawable, {
  } );
  
  return SVGSelfDrawable;
} );
