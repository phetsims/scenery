// Copyright 2002-2013, University of Colorado

/**
 * A unit that is drawable with a specific renderer
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Drawable = function Drawable( drawableState ) {
    
    // linked list handling
    this.previousDrawable = null;
    this.nextDrawable = null;
    
    this.drawableState = drawableState;
    this.renderer = drawableState.getDrawableRenderer();
    this.trail = drawableState.getTrail();
    this.transformBaseTrail = drawableState.getTransformBaseTrail();
    this.transformTrail = drawableState.getTransformTrail();
  };
  var Drawable = scenery.Drawable;
  
  inherit( Object, Drawable, {
    /*---------------------------------------------------------------------------*
    * DOM API
    *----------------------------------------------------------------------------*/
    
    // for DOM-renderer drawables
    getDomElement: function() {
      
    },
    
    /*---------------------------------------------------------------------------*
    * Canvas API
    *----------------------------------------------------------------------------*/
    
    paintCanvas: function( wrapper, offset ) {
      
    }
    
    /*---------------------------------------------------------------------------*
    * SVG API
    *----------------------------------------------------------------------------*/
    
    
    
  } );
  
  return Drawable;
} );
