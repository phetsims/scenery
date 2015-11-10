// Copyright 2013-2015, University of Colorado Boulder


/**
 * A linear gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  var inherit = require( 'PHET_CORE/inherit' );
  var Vector2 = require( 'DOT/Vector2' );
  var Gradient = require( 'SCENERY/util/Gradient' );

  // TODO: add the ability to specify the color-stops inline. possibly [ [0,color1], [0.5,color2], [1,color3] ]
  function LinearGradient( x0, y0, x1, y1 ) {
    assert && assert( isFinite( x0 ) && isFinite( y0 ) && isFinite( x1 ) && isFinite( y1 ) );
    var usesVectors = y1 === undefined;
    if ( usesVectors ) {
      assert && assert( ( x0 instanceof Vector2 ) && ( y0 instanceof Vector2 ), 'If less than 4 parameters are given, the first two parameters must be Vector2' );
    }
    this.start = usesVectors ? x0 : new Vector2( x0, y0 );
    this.end = usesVectors ? y0 : new Vector2( x1, y1 );

    // use the global scratch canvas instead of creating a new Canvas
    Gradient.call( this, scenery.scratchContext.createLinearGradient( x0, y0, x1, y1 ) );
  }
  scenery.register( 'LinearGradient', LinearGradient );

  inherit( Gradient, LinearGradient, {

    isLinearGradient: true,

    // seems we need the defs: http://stackoverflow.com/questions/7614209/linear-gradients-in-svg-without-defs
    // SVG: spreadMethod 'pad' 'reflect' 'repeat' - find Canvas usage
    getSVGDefinition: function() {
      /* Approximate example of what we are creating:
       <linearGradient id="grad2" x1="0" y1="0" x2="100" y2="0" gradientUnits="userSpaceOnUse">
       <stop offset="0" style="stop-color:rgb(255,255,0);stop-opacity:1" />
       <stop offset="0.5" style="stop-color:rgba(255,255,0,0);stop-opacity:0" />
       <stop offset="1" style="stop-color:rgb(255,0,0);stop-opacity:1" />
       </linearGradient>
       */
      var definition = document.createElementNS( scenery.svgns, 'linearGradient' );
      definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'x1', this.start.x );
      definition.setAttribute( 'y1', this.start.y );
      definition.setAttribute( 'x2', this.end.x );
      definition.setAttribute( 'y2', this.end.y );
      if ( this.transformMatrix ) {
        definition.setAttribute( 'gradientTransform', this.transformMatrix.getSVGTransform() );
      }

      _.each( this.stops, function( stop ) {
        var stopElement = document.createElementNS( scenery.svgns, 'stop' );
        stopElement.setAttribute( 'offset', stop.ratio );
        // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
        // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
        // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
        // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
        stopElement.setAttribute( 'style', 'stop-color: ' + stop.color.withAlpha( 1 ).toCSS() + '; stop-opacity: ' + stop.color.a.toFixed( 20 ) + ';' );
        definition.appendChild( stopElement );
      } );

      return definition;
    },

    toString: function() {
      var result = 'new scenery.LinearGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.end.x + ', ' + this.end.y + ' )';

      _.each( this.stops, function( stop ) {
        result += '.addColorStop( ' + stop.ratio + ', \'' + ( stop.color.toCSS ? stop.color.toCSS() : stop.color.toString() ) + '\' )';
      } );

      return result;
    }
  } );

  return LinearGradient;
} );
