// Copyright 2002-2013, University of Colorado

/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying any changes at a later time.
 *
 * Target API (responsible for the creation and rendering of the SVG element, along with sending feature-dirty flags):
 * {
 *   createSVGFragment: function() : SVGElement
 *   << TODO way of registering for state changes. allow target to control object with its own flags, so we don't need dynamic name lookups. (text could have state.font, state.text, etc.)
 *      need to send an event on transition from clean=>dirty so we can mark the change as needing to be updated >>
 *   << TODO way of unregistering for state changes >>
 *   << TODO way of applying state changes (we have the target track the defs changes, to which we provide access) >>
 * }
 * A common target is a Node that can be rendered in SVG, but we also (may) allow displaying Canvas caches with SVG (as an Image with a custom offset),
 * or possibly other things in the future
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.SVGDrawable = function SVGDrawable( trail, renderer, transformTrail, target ) {
    Drawable.call( this, trail, renderer, transformTrail );
    
    this.target = target;
  };
  var SVGDrawable = scenery.SVGDrawable;
  
  inherit( Drawable, SVGDrawable, {
  } );
  
  return SVGDrawable;
} );
