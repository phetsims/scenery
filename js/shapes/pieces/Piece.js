// Copyright 2002-2012, University of Colorado

/**
 * Represents a higher-level command for Shape, and generally mimics the Canvas drawing api.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Piece = {
    /*
     * Will contain (for pieces):
     * methods:
     * writeToContext( context ) - Executes this drawing command directly on a Canvas context
     * transformed( matrix )     - Returns a transformed copy of this piece
     * applyPiece( shape )       - Applies this piece to a shape, essentially internally executing the Canvas api and creating subpaths and segments.
     *                             This is necessary, since pieces like Rect can actually contain many more than one segment, and drawing pieces depends
     *                             on context / subpath state.
     */
  };
  var Piece = scenery.Piece;
  
  return Piece;
} );
