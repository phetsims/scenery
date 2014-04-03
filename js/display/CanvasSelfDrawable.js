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
  
  scenery.CanvasSelfDrawable = function CanvasSelfDrawable( trail, renderer, instance ) {
    Drawable.call( this, renderer );
    
    this.instance = instance;
  };
  var CanvasSelfDrawable = scenery.CanvasSelfDrawable;
  
  inherit( Drawable, CanvasSelfDrawable, {
    
    paintCanvas: function( wrapper, offset ) {
      // TODO: draw!
    }
    
  } );
  
  return CanvasSelfDrawable;
} );
