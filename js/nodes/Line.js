// Copyright 2002-2013, University of Colorado

/**
 * A line that inherits Path, and allows for optimized drawing,
 * and improved line handling.
 *
 * TODO: add DOM support
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var KiteLine = require( 'KITE/segments/Line' );
  
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Vector2 = require( 'DOT/Vector2' );
  
  /**
   * Currently, all numerical parameters should be finite.
   * x1:         x-position of the start
   * y1:         y-position of the start
   * x2:         x-position of the end
   * y2:         y-position of the end
   *
   * Available constructors:
   * new Line( x1, y1, x2, y2, { ... } )
   * new Line( new Vector2( x1, y1 ), new Vector2( x2, y2 ), { ... } )
   * new Line( { x1: x1, y1: y1, x2: x2, y2: y2,  ... } )
   */
  scenery.Line = function Line( x1, y1, x2, y2, options ) {
    if ( typeof x1 === 'object' ) {
      if ( x1 instanceof Vector2 ) {
        // assumes Line( Vector2, Vector2, options );
        this._x1 = x1.x;
        this._y1 = x1.y;
        this._x2 = y1.x;
        this._y2 = y1.y;
        options = x2 || {};
      } else {
        // assumes Line( { ... } ), init to zero for now
        this._x1 = 0;
        this._y1 = 0;
        this._x2 = 0;
        this._y2 = 0;
        options = x1 || {};
      }
    } else {
      // new Line(  x1, y1, x2, y2, [options] )
      this._x1 = x1;
      this._y1 = y1;
      this._x2 = x2;
      this._y2 = y2;
      
      // ensure we have a parameter object
      options = options || {};
    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation
    
    Path.call( this, null, options );
  };
  var Line = scenery.Line;
  
  inherit( Path, Line, {
    setLine: function( x1, y1, x2, y2 ) {
      assert && assert( x1 !== undefined && y1 !== undefined && x2 !== undefined && y2 !== undefined, 'parameters need to be defined' );
      
      this._x1 = x1;
      this._y1 = y1;
      this._x2 = x2;
      this._y2 = y2;
      this.invalidateLine();
    },
    
    setPoint1: function( x1, y1 ) {
      if ( typeof x1 === 'number' ) {
        // setPoint1( x1, y1 );
        this.setLine( x1, y1, this._x2, this._y2 );
      } else {
        // setPoint1( Vector2 )
        this.setLine( x1.x, x1.y, this._x2, this._y2 );
      }
    },
    set p1( point ) { this.setPoint1( point ); },
    get p1() { return new Vector2( this._x1, this._y1 ); },
    
    setPoint2: function( x2, y2 ) {
      if ( typeof x2 === 'number' ) {
        // setPoint2( x2, y2 );
        this.setLine( this._x1, this._y1, x2, y2 );
      } else {
        // setPoint2( Vector2 )
        this.setLine( this._x1, this._y1, x2.x, x2.y );
      }
    },
    set p2( point ) { this.setPoint2( point ); },
    get p2() { return new Vector2( this._x2, this._y2 ); },
    
    createLineShape: function() {
      return Shape.lineSegment( this._x1, this._y1, this._x2, this._y2 );
    },
    
    invalidateLine: function() {
      assert && assert( isFinite( this._x1 ), 'A rectangle needs to have a finite x1 (' + this._x1 + ')' );
      assert && assert( isFinite( this._y1 ), 'A rectangle needs to have a finite y1 (' + this._y1 + ')' );
      assert && assert( isFinite( this._x2 ), 'A rectangle needs to have a finite x2 (' + this._x2 + ')' );
      assert && assert( isFinite( this._y2 ), 'A rectangle needs to have a finite y2 (' + this._y2 + ')' );
      
      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;
      
      // should invalidate the path and ensure a redraw
      this.invalidateShape();
    },
    
    containsPointSelf: function( point ) {
      if ( this._strokePickable ) {
        return Path.prototype.containsPointSelf.call( this, point );
      } else {
        return false; // nothing is in a line! (although maybe we should handle edge points properly?)
      }
    },
    
    intersectsBoundsSelf: function( bounds ) {
      // TODO: optimization
      return new KiteLine( this.p1, this.p2 ).intersectsBounds( bounds );
    },
    
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      context.beginPath();
      context.moveTo( this._x1, this._y1 );
      context.lineTo( this._x2, this._y2 );
      context.closePath();
      
      if ( this._stroke ) {
        this.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.stroke();
        this.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },
    
    computeShapeBounds: function() {
      return Path.prototype.computeShapeBounds.call( this );
    },
    
    // create a rect instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'line' );
    },
    
    // optimized for the rect element instead of path
    updateSVGFragment: function( rect ) {
      // see http://www.w3.org/TR/SVG/shapes.html#LineElement
      rect.setAttribute( 'x1', this._x1 );
      rect.setAttribute( 'y1', this._y1 );
      rect.setAttribute( 'x2', this._x2 );
      rect.setAttribute( 'y2', this._y2 );
      
      rect.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Line( ' + this._x1 + ', ' + this._y1 + ', ' + this._x1 + ', ' + this._y1 + ', {' + propLines + '} )';
    },
    
    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Line to something non-null' );
      } else {
        // probably called from the Path constructor
        this.invalidateShape();
      }
    },
    
    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createLineShape();
      }
      return this._shape;
    },
    
    hasShape: function() {
      return true;
    }
    
  } );
  
  function addLineProp( capitalizedShort ) {
    var lowerShort = capitalizedShort.toLowerCase();
    
    var getName = 'get' + capitalizedShort;
    var setName = 'set' + capitalizedShort;
    var privateName = '_' + lowerShort;
    
    Line.prototype[getName] = function() {
      return this[privateName];
    };
    
    Line.prototype[setName] = function( value ) {
      if ( this[privateName] !== value ) {
        this[privateName] = value;
        this.invalidateLine();
      }
      return this;
    };
    
    Object.defineProperty( Line.prototype, lowerShort, {
      set: Line.prototype[setName],
      get: Line.prototype[getName]
    } );
  }
  
  addLineProp( 'X1' );
  addLineProp( 'Y1' );
  addLineProp( 'X2' );
  addLineProp( 'Y2' );
  
  // not adding mutators for now
  Line.prototype._mutatorKeys = [ 'p1', 'p2', 'x1', 'y1', 'x2', 'y2' ].concat( Path.prototype._mutatorKeys );
  
  return Line;
} );


