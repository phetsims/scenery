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
  
  scenery.CanvasBlock = function CanvasBlock( renderer ) {
    Drawable.call( this, renderer );
    // TODO: add count of boundsless objects
    // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)
  };
  var CanvasBlock = scenery.CanvasBlock;
  
  inherit( Drawable, CanvasBlock, {
  } );
  
  return CanvasBlock;
} );
