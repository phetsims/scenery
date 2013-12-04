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
  var Block = require( 'SCENERY/display/Block' );
  
  scenery.CanvasBlock = function CanvasBlock() {
    // TODO: add count of boundsless objects
  };
  var CanvasBlock = scenery.CanvasBlock;
  
  inherit( Block, CanvasBlock, {
    getDomElement: function() {
      
    }
    
    // TODO: add canvas draw methods
  } );
  
  return CanvasBlock;
} );
