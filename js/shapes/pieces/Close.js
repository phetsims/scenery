// Copyright 2002-2012, University of Colorado

/**
 * Closes a subpath
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', true );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Piece = require( 'SCENERY/shapes/pieces/Piece' );
  var Subpath = require( 'SCENERY/shapes/util/Subpath' );
  
  Piece.Close = function() {};
  Piece.Close.prototype = {
    constructor: Piece.Close,
    
    writeToContext: function( context ) {
      context.closePath();
    },
    
    transformed: function( matrix ) {
      return [this];
    },
    
    applyPiece: function( shape ) {
      if ( shape.hasSubpaths() ) {
        var previousPath = shape.getLastSubpath();
        var nextPath = new Subpath();
        
        previousPath.close();
        shape.addSubpath( nextPath );
        nextPath.addPoint( previousPath.getFirstPoint() );
      }
    }
  };
  
  return Piece.Close;
} );
