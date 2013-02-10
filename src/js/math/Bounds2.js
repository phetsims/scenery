// Copyright 2002-2012, University of Colorado

/**
 * An immutable rectangle-shaped bounded area (bounding box) in 2D
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

// ensure proper namespace
var phet = phet || {};
phet.math = phet.math || {};

// create a new scope
(function () {
  "use strict";
  
  var Vector2 = phet.math.Vector2;
  
  // not using x,y,width,height so that it can handle infinity-based cases in a better way
  phet.math.Bounds2 = function( minX, minY, maxX, maxY ) {
    this.minX = minX;
    this.minY = minY;
    this.maxX = maxX;
    this.maxY = maxY;
  };

  var Bounds2 = phet.math.Bounds2;

  Bounds2.prototype = {
    constructor: Bounds2,
    
    // properties of this bounding box
    width: function() { return this.maxX - this.minX; },
    height: function() { return this.maxY - this.minY; },
    x: function() { return this.minX; },
    y: function() { return this.minY; },
    centerX: function() { return ( this.maxX + this.minX ) / 2; },
    centerY: function() { return ( this.maxY + this.minY ) / 2; },
    isEmpty: function() { return this.width() <= 0 || this.height() <= 0; },
    
    // immutable operations (bounding-box style handling, so that the relevant bounds contain everything)
    union: function( other ) {
      return new Bounds2(
        Math.min( this.minX, other.minX ),
        Math.min( this.minY, other.minY ),
        Math.max( this.maxX, other.maxX ),
        Math.max( this.maxY, other.maxY )
      );
    },
    intersection: function( other ) {
      return new Bounds2(
        Math.max( this.minX, other.minX ),
        Math.max( this.minY, other.minY ),
        Math.min( this.maxX, other.maxX ),
        Math.min( this.maxY, other.maxY )
      );
    },
    // TODO: difference should be well-defined, but more logic is needed to compute
    
    // like a union with a point-sized bounding box
    withPoint: function( point ) {
      return new Bounds2(
        Math.min( this.minX, point.x ),
        Math.min( this.minY, point.y ),
        Math.max( this.maxX, point.x ),
        Math.max( this.maxY, point.y )
      );
    },
    
    withMinX: function( minX ) { return new Bounds2( minX, this.minY, this.maxX, this.maxY ); },
    withMinY: function( minY ) { return new Bounds2( this.minX, minY, this.maxX, this.maxY ); },
    withMaxX: function( maxX ) { return new Bounds2( this.minX, this.minY, maxX, this.maxY ); },
    withMaxY: function( maxY ) { return new Bounds2( this.minX, this.minY, this.maxX, maxY ); },
    
    // copy rounded to integral values, expanding where necessary
    roundedOut: function() {
      return new Bounds2(
        Math.floor( this.minX ),
        Math.floor( this.minY ),
        Math.ceil( this.maxX ),
        Math.ceil( this.maxY )
      );
    },
    
    // copy rounded to integral values, contracting where necessary
    roundedIn: function() {
      return new Bounds2(
        Math.ceil( this.minX ),
        Math.ceil( this.minY ),
        Math.floor( this.maxX ),
        Math.floor( this.maxY )
      );
    },
    
    // whether the point is inside the bounding box
    containsPoint: function( point ) {
      return this.minX <= point.x && point.x <= this.maxX && this.minY <= point.y && point.y <= this.maxY;
    },
    
    // whether this bounding box completely contains the argument bounding box
    containsBounds: function( bounds ) {
      return this.minX <= bounds.minX && this.maxX >= bounds.maxX && this.minY <= bounds.minY && this.maxY >= bounds.maxY;
    },
    
    intersectsBounds: function( bounds ) {
      // TODO: more efficient way of doing this?
      return !this.intersection( bounds ).isEmpty();
    },
    
    // transform a bounding box.
    // NOTE that box.transformed( matrix ).transformed( inverse ) may be larger than the original box
    transformed: function( matrix ) {
      if ( this.isEmpty() ) {
        return Bounds2.NOTHING;
      }
      var result = Bounds2.NOTHING;
      
      // make sure all 4 corners are inside this transformed bounding box
      result = result.withPoint( matrix.timesVector2( new Vector2( this.minX, this.minY ) ) );
      result = result.withPoint( matrix.timesVector2( new Vector2( this.minX, this.maxY ) ) );
      result = result.withPoint( matrix.timesVector2( new Vector2( this.maxX, this.minY ) ) );
      result = result.withPoint( matrix.timesVector2( new Vector2( this.maxX, this.maxY ) ) );
      return result;
    },
    
    // returns copy expanded on all sides by length d
    dilated: function( d ) {
      return new Bounds2( this.minX - d, this.minY - d, this.maxX + d, this.maxY + d );
    },
    
    // returns copy contracted on all sides by length d
    eroded: function( d ) {
      return this.dilated( -d );
    },

    toString: function () {
      return '[x:(' + this.minX + ',' + this.maxX + '),y:(' + this.minY + ',' + this.maxY + ')]';
    },

    equals: function ( other ) {
      return this.minX === other.minX && this.minY === other.minY && this.maxX === other.maxX && this.maxY === other.maxY;
    }
  };
  
  // specific bounds useful for operations
  Bounds2.EVERYTHING = new Bounds2( Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY );
  Bounds2.NOTHING = new Bounds2( Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY );
})();
