// Copyright 2002-2013, University of Colorado

/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.SVGBlock = function SVGBlock( renderer ) {
    Drawable.call( this, renderer );
    // TODO: add count of boundsless objects
  };
  var SVGBlock = scenery.SVGBlock;
  
  inherit( Drawable, SVGBlock, {
    getDomElement: function() {
      
    }
    
    // TODO: add canvas draw methods
  } );
  
  return SVGBlock;
} );
