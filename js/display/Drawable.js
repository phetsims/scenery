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
  
  scenery.Drawable = function Drawable( trail, renderer, transformTrail ) {
    // what block we are being rendered into (will be filled in later)
    this.block = null;
    
    // linked list handling (will be filled in later)
    this.previousDrawable = null;
    this.nextDrawable = null;
    
    this.trail = trail;
    this.renderer = renderer;
    this.transformTrail = transformTrail;
  };
  var Drawable = scenery.Drawable;
  
  inherit( Object, Drawable, {
    dispose: function() {
      
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
