// Copyright 2002-2013, University of Colorado

/**
 * A generic display block (TODO more docs once finished)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Block = function Block() {
    this.parentBlock = null;
    this.childrenBlocks = [];
    
    this.previousBlock = null;
    this.nextBlock = null;
  };
  var Block = scenery.Block;
  
  // API:
  inherit( Object, Block, {
    
    // for direct DOM / backbone blocks, anything that supports a direct DOM element output
    getDomElement: function() {
      
    }
  } );
  
  return Block;
} );
