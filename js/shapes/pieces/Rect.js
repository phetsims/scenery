// Copyright 2002-2012, University of Colorado

/**
 * Draws an arc.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  require( 'SCENERY/shapes/pieces/MoveTo' );
  require( 'SCENERY/shapes/pieces/Close' );
  var Subpath = require( 'SCENERY/shapes/util/Subpath' );
  var Line = require( 'SCENERY/shapes/segments/Line' );
  
  // for brevity
  function p( x,y ) { return new Vector2( x, y ); }
  
  Piece.Rect = function( upperLeft, lowerRight ) {
    this.upperLeft = upperLeft;
    this.lowerRight = lowerRight;
    this.x = this.upperLeft.x;
    this.y = this.upperLeft.y;
    this.width = this.lowerRight.x - this.x;
    this.height = this.lowerRight.y - this.y;
  };
  Piece.Rect.prototype = {
    constructor: Piece.Rect,
    
    writeToContext: function( context ) {
      context.rect( this.x, this.y, this.width, this.height );
    },
    
    transformed: function( matrix ) {
      var a = matrix.timesVector2( p( this.x, this.y ) );
      var b = matrix.timesVector2( p( this.x + this.width, this.y ) );
      var c = matrix.timesVector2( p( this.x + this.width, this.y + this.height ) );
      var d = matrix.timesVector2( p( this.x, this.y + this.height ) );
      return [new Piece.MoveTo( a ), new Piece.LineTo( b ), new Piece.LineTo( c ), new Piece.LineTo( d ), new Piece.Close(), new Piece.MoveTo( a )];
    },
    
    applyPiece: function( shape ) {
      var subpath = new Subpath();
      shape.addSubpath( subpath );
      subpath.addPoint( p( this.x, this.y ) );
      subpath.addPoint( p( this.x + this.width, this.y ) );
      subpath.addPoint( p( this.x + this.width, this.y + this.height ) );
      subpath.addPoint( p( this.x, this.y + this.height ) );
      subpath.addSegment( new Line( subpath.points[0], subpath.points[1] ) );
      subpath.addSegment( new Line( subpath.points[1], subpath.points[2] ) );
      subpath.addSegment( new Line( subpath.points[2], subpath.points[3] ) );
      subpath.close();
      shape.addSubpath( new Subpath() );
      shape.getLastSubpath().addPoint( p( this.x, this.y ) );
      shape.bounds = shape.bounds.withPoint( this.upperLeft ).withPoint( this.lowerRight );
      assert && assert( !isNaN( shape.bounds.x() ) );
    }
  };
  
  return Piece.Rect;
} );
