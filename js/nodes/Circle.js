// Copyright 2002-2012, University of Colorado

/**
 * A circular node that inherits Path, and allows for optimized drawing,
 * and improved parameter handling.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  
  scenery.Circle = function Circle( radius, options ) {
    if ( typeof radius === 'object' ) {
      // allow new Circle( { radius: ... } )
      // the mutators will call invalidateCircle() and properly set the shape
      options = radius;
    } else {
      this._radius = radius;
      
      // ensure we have a parameter object
      options = options || {};
      
      // fallback for non-canvas or non-svg rendering, and for proper bounds computation
      options.shape = Shape.circle( 0, 0, radius );
    }
    
    Path.call( this, options );
  };
  var Circle = scenery.Circle;
  
  inherit( Circle, Path, {
    invalidateCircle: function() {
      // setShape should invalidate the path and ensure a redraw
      this.setShape( Shape.circle( 0, 0, this._radius ) );
    },
    
    // create a circle instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'circle' );
    },
    
    // optimized for the circle element instead of path
    updateSVGFragment: function( circle ) {
      circle.setAttribute( 'r', this._radius );
      
      circle.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Circle( ' + this._radius + ', {' + propLines + '} )';
    },
    
    getRadius: function() {
      return this._radius;
    },
    
    setRadius: function( radius ) {
      if ( this._radius !== radius ) {
        this._radius = radius;
        this.invalidateCircle();
      }
      return this;
    },
    
    get radius() { return this.getRadius(); },
    set radius( value ) { return this.setRadius( value ); }
  } );
  
  // not adding mutators for now
  Circle.prototype._mutatorKeys = [ 'radius' ].concat( Path.prototype._mutatorKeys );
  
  return Circle;
} );
