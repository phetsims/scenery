// Copyright 2002-2012, University of Colorado

/**
 * A radial gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Vector2 = require( 'DOT/Vector2' );
  
  // TODO: support canvas method as base, map to SVG method with color-stop scaling
  scenery.RadialGradient = function( x0, y0, r0, x1, y1, r1 ) {
    throw new Error( 'RadialGradient not implemented' );
  };
  var RadialGradient = scenery.RadialGradient;
  
  RadialGradient.prototype = {
    constructor: RadialGradient,
    
    addColorStop: function( ratio, color ) {
      throw new Error( 'RadialGradient.addColorStop not implemented' );
    },
    
    getCanvasStyle: function() {
      throw new Error( 'RadialGradient.getCanvasStyle not implemented' );
    },
    
    getSVGDefinition: function( id ) {
      throw new Error( 'RadialGradient.getSVGDefinition not implemented' );
    }
  };
  
  return RadialGradient;
} );
