// Copyright 2002-2012, University of Colorado

/**
 * Styles needed to determine a stroked line shape.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  var assertExtra = require( 'ASSERT/assert' )( 'scenery.extra', true );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.LineStyles = function( args ) {
    if ( args === undefined ) {
      args = {};
    }
    this.lineWidth = args.lineWidth !== undefined ? args.lineWidth : 1;
    this.lineCap = args.lineCap !== undefined ? args.lineCap : 'butt'; // butt, round, square
    this.lineJoin = args.lineJoin !== undefined ? args.lineJoin : 'miter'; // miter, round, bevel
    this.miterLimit = args.miterLimit !== undefined ? args.miterLimit : 10; // see https://svgwg.org/svg2-draft/painting.html for miterLimit computations
  };
  var LineStyles = scenery.LineStyles;
  LineStyles.prototype = {
    constructor: LineStyles,
    
    equals: function( other ) {
      return this.lineWidth === other.lineWidth && this.lineCap === other.lineCap && this.lineJoin === other.lineJoin && this.miterLimit === other.miterLimit;
    }
  };
  
  return scenery.LineStyles;
} );
