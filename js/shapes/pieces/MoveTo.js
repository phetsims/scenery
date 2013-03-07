// Copyright 2002-2012, University of Colorado

/**
 * Creates a new subpath starting at the specified point
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  var Subpath = require( 'SCENERY/shapes/util/Subpath' );
  
  Piece.MoveTo = function( point ) {
    this.point = point;
  };
  Piece.MoveTo.prototype = {
    constructor: Piece.MoveTo,
    
    writeToContext: function( context ) {
      context.moveTo( this.point.x, this.point.y );
    },
    
    transformed: function( matrix ) {
      return [new Piece.MoveTo( matrix.timesVector2( this.point ) )];
    },
    
    applyPiece: function( shape ) {
      var subpath = new Subpath();
      subpath.addPoint( this.point );
      shape.addSubpath( subpath );
    }
  };
  
  return Piece.MoveTo;
} );
