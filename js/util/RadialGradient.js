// Copyright 2013-2015, University of Colorado Boulder


/**
 * A radial gradient that can be passed into the 'fill' or 'stroke' parameters.
 *
 * SVG gradients, see http://www.w3.org/TR/SVG/pservers.html
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );

  var inherit = require( 'PHET_CORE/inherit' );
  var platform = require( 'PHET_CORE/platform' );
  var Vector2 = require( 'DOT/Vector2' );
  var Gradient = require( 'SCENERY/util/Gradient' );

  // TODO: support Vector2s for p0 and p1
  function RadialGradient( x0, y0, r0, x1, y1, r1 ) {
    this.start = new Vector2( x0, y0 );
    this.end = new Vector2( x1, y1 );

    // If we are using Safari, we need to work around incorrect gradient handling for now,
    // see https://github.com/phetsims/sun/issues/526
    if ( platform.safari ) {
      var x = ( x0 + x1 ) / 2;
      var y = ( y0 + y1 ) / 2;
      this.start.x = x;
      this.start.y = y;
      this.end.x = x;
      this.end.y = y;
    }

    // @public {number}
    this.startRadius = r0;
    this.endRadius = r1;

    // linear function from radius to point on the line from start to end
    this.focalPoint = this.start.plus( this.end.minus( this.start ).times( this.startRadius / ( this.startRadius - this.endRadius ) ) );

    // make sure that the focal point is in both circles. SVG doesn't support rendering outside of them
    if ( this.startRadius >= this.endRadius ) {
      assert && assert( this.focalPoint.minus( this.start ).magnitude() <= this.startRadius );
    }
    else {
      assert && assert( this.focalPoint.minus( this.end ).magnitude() <= this.endRadius );
    }

    // use the global scratch canvas instead of creating a new Canvas
    Gradient.call( this, scenery.scratchContext.createRadialGradient( x0, y0, r0, x1, y1, r1 ) );
  }

  scenery.register( 'RadialGradient', RadialGradient );

  inherit( Gradient, RadialGradient, {

    isRadialGradient: true,

    getSVGDefinition: function() {
      var startIsLarger = this.startRadius > this.endRadius;
      var largePoint = startIsLarger ? this.start : this.end;
      // var smallPoint = startIsLarger ? this.end : this.start;
      var maxRadius = Math.max( this.startRadius, this.endRadius );
      var minRadius = Math.min( this.startRadius, this.endRadius );

      var definition = document.createElementNS( scenery.svgns, 'radialGradient' );

      definition.setAttribute( 'gradientUnits', 'userSpaceOnUse' ); // so we don't depend on the bounds of the object being drawn with the gradient
      definition.setAttribute( 'cx', largePoint.x );
      definition.setAttribute( 'cy', largePoint.y );
      definition.setAttribute( 'r', maxRadius );
      definition.setAttribute( 'fx', this.focalPoint.x );
      definition.setAttribute( 'fy', this.focalPoint.y );
      if ( this.transformMatrix ) {
        definition.setAttribute( 'gradientTransform', this.transformMatrix.getSVGTransform() );
      }

      //TODO: replace with dot.Util.linear
      // maps x linearly from [a0,b0] => [a1,b1]
      function linearMap( a0, b0, a1, b1, x ) {
        return a1 + ( x - a0 ) * ( b1 - a1 ) / ( b0 - a0 );
      }

      function applyStop( stop ) {
        // flip the stops if the start has a larger radius
        var ratio = startIsLarger ? 1 - stop.ratio : stop.ratio;

        // scale the stops properly if the smaller radius isn't 0
        if ( minRadius > 0 ) {
          // scales our ratio from [0,1] => [minRadius/maxRadius,0]
          ratio = linearMap( 0, 1, minRadius / maxRadius, 1, ratio );
        }

        // TODO: store color in our stops array, so we don't have to create additional objects every time?
        var stopElement = document.createElementNS( scenery.svgns, 'stop' );
        stopElement.setAttribute( 'offset', ratio );
        // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
        // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
        // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
        // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
        stopElement.setAttribute( 'style', 'stop-color: ' + stop.color.withAlpha( 1 ).toCSS() + '; stop-opacity: ' + stop.color.a.toFixed( 20 ) + ';' );
        definition.appendChild( stopElement );
      }

      var i;
      // switch the direction we apply stops in, so that the ratios always are increasing.
      if ( startIsLarger ) {
        for ( i = this.stops.length - 1; i >= 0; i-- ) {
          applyStop( this.stops[ i ] );
        }
      }
      else {
        for ( i = 0; i < this.stops.length; i++ ) {
          applyStop( this.stops[ i ] );
        }
      }

      return definition;
    },

    toString: function() {
      var result = 'new scenery.RadialGradient( ' + this.start.x + ', ' + this.start.y + ', ' + this.startRadius + ', ' + this.end.x + ', ' + this.end.y + ', ' + this.endRadius + ' )';

      _.each( this.stops, function( stop ) {
        result += '.addColorStop( ' + stop.ratio + ', \'' + stop.color.toString() + '\' )';
      } );

      return result;
    }
  } );

  return RadialGradient;
} );
