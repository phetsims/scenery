// Copyright 2002-2013, University of Colorado

/**
 * A rectangular node that inherits Path, and allows for optimized drawing,
 * and improved rectangle handling.
 *
 * TODO: add DOM support
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  
  var Path = require( 'SCENERY/nodes/Path' );
  var Shape = require( 'KITE/Shape' );
  var Vector2 = require( 'DOT/Vector2' );
  
  /**
   * Currently, all numerical parameters should be finite.
   * x:         x-position of the upper-left corner (left bound)
   * y:         y-position of the upper-left corner (top bound)
   * width:     width of the rectangle to the right of the upper-left corner, required to be >= 0
   * height:    height of the rectangle below the upper-left corner, required to be >= 0
   * arcWidth:  positive width of the rounded corner, or 0 to indicate the corner should be sharp
   * arcHeight: positive height of the rounded corner, or 0 to indicate the corner should be sharp
   */
  scenery.Rectangle = function Rectangle( x, y, width, height, arcWidth, arcHeight, options ) {
    if ( typeof x === 'object' ) {
      // allow new Rectangle( { rectX: x, rectY: y, rectWidth: width, rectHeight: height, ... } )
      // the mutators will call invalidateRectangle() and properly set the shape
      options = x;
    } else if ( arguments.length < 6 ) {
      // new Rectangle( x, y, width, height, [options] )
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = 0;
      this._rectArcHeight = 0;
      
      // ensure we have a parameter object
      options = arcWidth || {};
      
      // fallback for non-canvas or non-svg rendering, and for proper bounds computation
      options.shape = this.createRectangleShape();
    } else {
      // normal case with args (including arcWidth / arcHeight)
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = arcWidth;
      this._rectArcHeight = arcHeight;
      
      // ensure we have a parameter object
      options = options || {};
      
      // fallback for non-canvas or non-svg rendering, and for proper bounds computation
      options.shape = this.createRectangleShape();
    }
    
    Path.call( this, options );
  };
  var Rectangle = scenery.Rectangle;
  
  inherit( Path, Rectangle, {
    setRect: function( x, y, width, height, arcWidth, arcHeight ) {
      sceneryAssert && sceneryAssert( x !== undefined && y !== undefined && width !== undefined && height !== undefined, 'x/y/width/height need to be defined' );
      
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = arcWidth || 0;
      this._rectArcHeight = arcHeight || 0;
      this.invalidateRectangle();
    },
    
    isRounded: function() {
      return this._rectArcWidth !== 0 && this._rectArcHeight !== 0;
    },
    
    createRectangleShape: function() {
      sceneryAssert && sceneryAssert( isFinite( this._rectX ), 'A rectangle needs to have a finite x (' + this._rectX + ')' );
      sceneryAssert && sceneryAssert( isFinite( this._rectY ), 'A rectangle needs to have a finite x (' + this._rectY + ')' );
      sceneryAssert && sceneryAssert( this._rectWidth >= 0 && isFinite( this._rectWidth ),
                                      'A rectangle needs to have a non-negative finite width (' + this._rectWidth + ')' );
      sceneryAssert && sceneryAssert( this._rectHeight >= 0 && isFinite( this._rectHeight ),
                                      'A rectangle needs to have a non-negative finite height (' + this._rectHeight + ')' );
      sceneryAssert && sceneryAssert( this._rectArcWidth >= 0 && isFinite( this._rectArcWidth ),
                                      'A rectangle needs to have a non-negative finite arcWidth (' + this._rectArcWidth + ')' );
      sceneryAssert && sceneryAssert( this._rectArcHeight >= 0 && isFinite( this._rectArcHeight ),
                                      'A rectangle needs to have a non-negative finite arcHeight (' + this._rectArcHeight + ')' );
      // sceneryAssert && sceneryAssert( !this.isRounded() || ( this._rectWidth >= this._rectArcWidth * 2 && this._rectHeight >= this._rectArcHeight * 2 ),
      //                                 'The rounded sections of the rectangle should not intersect (the length of the straight sections shouldn\'t be negative' );
      
      if ( this.isRounded() ) {
        // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
        var maximumArcSize = Math.min( this._rectWidth / 2, this._rectHeight / 2 );
        return Shape.roundRectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight,
                                     Math.min( maximumArcSize, this._rectArcWidth ), Math.min( maximumArcSize, this._rectArcHeight ) );
      } else {
        return Shape.rectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
      }
    },
    
    invalidateRectangle: function() {
      // setShape should invalidate the path and ensure a redraw
      this.setShape( this.createRectangleShape() );
    },
    
    computeShapeBounds: function() {
      // optimization, where we know our computed bounds will be just expanded by half the lineWidth if we are stroked (don't have to compute the stroke shape)
      return this._stroke ? this._shape.bounds.dilated( this._lineDrawingStyles.lineWidth / 2 ) : this._shape.bounds;
    },
    
    // accelerated hit detection for axis-aligned optionally-rounded rectangle
    // fast computation if it isn't rounded. if rounded, we check if a corner computation is needed (usually isn't), and only check that one needed corner
    containsPointSelf: function( point ) {
      var result = point.x >= this._rectX &&
                   point.x <= this._rectX + this._rectWidth &&
                   point.y >= this._rectY &&
                   point.y <= this._rectY + this._rectHeight;
      
      if ( !result || !this.isRounded() ) {
        return result;
      }
      
      // we are rounded and inside the logical rectangle (if it didn't have rounded corners)
      
      // closest corner arc's center (we assume the rounded rectangle's arcs are 90 degrees fully, and don't intersect)
      var closestCornerX, closestCornerY, guaranteedInside = false;
      
      // if we are to the inside of the closest corner arc's center, we are guaranteed to be in the rounded rectangle (guaranteedInside)
      if ( point.x < this._rectX + this._rectWidth / 2 ) {
        closestCornerX = this._rectX + this._rectArcWidth;
        guaranteedInside = guaranteedInside || point.x >= closestCornerX;
      } else {
        closestCornerX = this._rectX + this._rectWidth - this._rectArcWidth;
        guaranteedInside = guaranteedInside || point.x <= closestCornerX;
      }
      if ( guaranteedInside ) { return true; }
      
      if ( point.y < this._rectY + this._rectHeight / 2 ) {
        closestCornerY = this._rectY + this._rectArcHeight;
        guaranteedInside = guaranteedInside || point.y >= closestCornerY;
      } else {
        closestCornerY = this._rectY + this._rectHeight - this._rectArcHeight;
        guaranteedInside = guaranteedInside || point.y <= closestCornerY;
      }
      if ( guaranteedInside ) { return true; }
      
      // we are now in the rectangular region between the logical corner and the center of the closest corner's arc.
      
      // offset from the closest corner's arc center
      var offsetX = point.x - closestCornerX;
      var offsetY = point.y - closestCornerY;
      
      // normalize the coordinates so now we are dealing with a unit circle
      // (technically arc, but we are guaranteed to be in the area covered by the arc, so we just consider the circle)
      // NOTE: we are rounded, so both arcWidth and arcHeight are non-zero (this is well defined)
      offsetX /= this._rectArcWidth;
      offsetY /= this._rectArcHeight;
      
      offsetX *= offsetX;
      offsetY *= offsetY;
      return offsetX + offsetY <= 1; // return whether we are in the rounded corner. see the formula for an ellipse
    },
    
    // override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
      if ( this.isRounded() ) {
        return Path.prototype.paintCanvas.call( this, wrapper );
      }
      
      // TODO: how to handle fill/stroke delay optimizations here?
      if ( this._fill ) {
        this.beforeCanvasFill( wrapper ); // defined in Fillable
        context.fillRect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
        this.afterCanvasFill( wrapper ); // defined in Fillable
      }
      if ( this._stroke ) {
        this.beforeCanvasStroke( wrapper ); // defined in Strokable
        context.strokeRect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
        this.afterCanvasStroke( wrapper ); // defined in Strokable
      }
    },
    
    // create a rect instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( 'http://www.w3.org/2000/svg', 'rect' );
    },
    
    // optimized for the rect element instead of path
    updateSVGFragment: function( rect ) {
      // see http://www.w3.org/TR/SVG/shapes.html#RectElement
      rect.setAttribute( 'x', this._rectX );
      rect.setAttribute( 'y', this._rectY );
      rect.setAttribute( 'width', this._rectWidth );
      rect.setAttribute( 'height', this._rectHeight );
      rect.setAttribute( 'rx', this._rectArcWidth );
      rect.setAttribute( 'ry', this._rectArcHeight );
      
      rect.setAttribute( 'style', this.getSVGFillStyle() + this.getSVGStrokeStyle() );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Rectangle( ' + this._rectX + ', ' + this._rectY + ', ' + 
                                         this._rectWidth + ', ' + this._rectHeight + ', ' +
                                         this._rectArcWidth + ', ' + this._rectArcHeight + ', {' + propLines + '} )';
    }
    
  } );
  
  function addRectProp( capitalizedShort ) {
    var getName = 'getRect' + capitalizedShort;
    var setName = 'setRect' + capitalizedShort;
    var privateName = '_rect' + capitalizedShort;
    
    Rectangle.prototype[getName] = function() {
      return this[privateName];
    };
    
    Rectangle.prototype[setName] = function( value ) {
      this[privateName] = value;
      this.invalidateRectangle();
      return this;
    };
    
    Object.defineProperty( Rectangle.prototype, 'rect' + capitalizedShort, {
      set: Rectangle.prototype[setName],
      get: Rectangle.prototype[getName]
    } );
  }
  
  addRectProp( 'X' );
  addRectProp( 'Y' );
  addRectProp( 'Width' );
  addRectProp( 'Height' );
  addRectProp( 'ArcWidth' );
  addRectProp( 'ArcHeight' );
  
  // not adding mutators for now
  Rectangle.prototype._mutatorKeys = [ 'rectX', 'rectY', 'rectWidth', 'rectHeight', 'rectArcWidth', 'rectArcHeight' ].concat( Path.prototype._mutatorKeys );
  
  return Rectangle;
} );


