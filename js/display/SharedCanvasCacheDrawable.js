// Copyright 2002-2013, University of Colorado

/**
 * TODO docs
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.SharedCanvasCacheDrawable = function SharedCanvasCacheDrawable( trail, renderer, instance, sharedInstance ) {
    Drawable.call( this, renderer );
    
    // TODO: NOTE: may have to separate into separate drawables for separate group renderers
    
    this.instance = instance;
    this.sharedInstance = sharedInstance;
  };
  var SharedCanvasCacheDrawable = scenery.SharedCanvasCacheDrawable;
  
  inherit( Drawable, SharedCanvasCacheDrawable, {
    // TODO: support Canvas/SVG/DOM
  } );
  
  return SharedCanvasCacheDrawable;
} );
