//  Copyright 2002-2014, University of Colorado Boulder

/**
 *
 * Data Structure used in  bin-packing algorithm. see Packer
 *
 * @author Sharfudeen Ashraf
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );


  /**
   *
   * @param x
   * @param y
   * @constructor
   */
  function BlockItem( x, y, width, height ) {
    this.x = x;
    this.y = y;
    this.width = width;
    this.height = height;
    this.used = false;
    this.position = null; //the starting position of this block within a spriteSheet
    this.right = null;//the BlockItem right to "this" item
    this.down = null;//the BlockItem down to "this" item
  }

  return inherit( Object, BlockItem );

} );
