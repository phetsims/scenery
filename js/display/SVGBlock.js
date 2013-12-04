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
  
  scenery.SVGBlock = function SVGBlock() {
    // TODO: add count of boundsless objects
  };
  var SVGBlock = scenery.SVGBlock;
  
  inherit( Block, SVGBlock, {
    getDomElement: function() {
      
    }
    
    // TODO: add canvas draw methods
  } );
  
  return SVGBlock;
} );
