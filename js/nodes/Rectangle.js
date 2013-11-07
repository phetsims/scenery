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
  var Bounds2 = require( 'DOT/Bounds2' );
  
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
      if ( x instanceof Bounds2 ) {
        // allow new Rectangle( bounds2, { ... } ) or new Rectangle( bounds2, arcWidth, arcHeight, options )
        this._rectX = x.minX;
        this._rectY = x.minY;
        this._rectWidth = x.width;
        this._rectHeight = x.height;
        if ( arguments.length < 3 ) {
          // Rectangle( bounds2, { ... } )
          options = y;
          this._rectArcWidth = 0;
          this._rectArcHeight = 0;
        } else {
          // Rectangle( bounds2, arcWidth, arcHeight, { ... } )
          options = height;
          this._rectArcWidth = y;
          this._rectArcHeight = width;
        }
      } else {
        // allow new Rectangle( { rectX: x, rectY: y, rectWidth: width, rectHeight: height, ... } )
        // the mutators will call invalidateRectangle() and properly set the shape
        options = x;
        this._rectX = options.rectX || 0;
        this._rectY = options.rectY || 0;
        this._rectWidth = options.rectWidth;
        this._rectHeight = options.rectHeight;
        this._rectArcWidth = options.rectArcWidth || 0;
        this._rectArcHeight = options.rectArcHeight || 0;
      }
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
      
    }
    // fallback for non-canvas or non-svg rendering, and for proper bounds computation

    Path.call( this, null, options );
  };
  var Rectangle = scenery.Rectangle;
  
  inherit( Path, Rectangle, {
    setRect: function( x, y, width, height, arcWidth, arcHeight ) {
      assert && assert( x !== undefined && y !== undefined && width !== undefined && height !== undefined, 'x/y/width/height need to be defined' );
      
      this._rectX = x;
      this._rectY = y;
      this._rectWidth = width;
      this._rectHeight = height;
      this._rectArcWidth = arcWidth || 0;
      this._rectArcHeight = arcHeight || 0;
      this.invalidateRectangle();
    },
    
    setRectBounds: function( bounds ) {
      this.setRect( bounds.x, bounds.y, bounds.width, bounds.height );
    },
    
    isRounded: function() {
      return this._rectArcWidth !== 0 && this._rectArcHeight !== 0;
    },
    
    computeShapeBounds: function() {
      var bounds = new Bounds2( this._rectX, this._rectY, this._rectX + this._rectWidth, this._rectY + this._rectHeight );
      if ( this._stroke ) {
        // since we are axis-aligned, any stroke will expand our bounds by a guaranteed set amount
        bounds = bounds.dilated( this.getLineWidth() / 2 );
      }
      return bounds;
    },
    
    createRectangleShape: function() {
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
      assert && assert( isFinite( this._rectX ), 'A rectangle needs to have a finite x (' + this._rectX + ')' );
      assert && assert( isFinite( this._rectY ), 'A rectangle needs to have a finite x (' + this._rectY + ')' );
      assert && assert( this._rectWidth >= 0 && isFinite( this._rectWidth ),
                                      'A rectangle needs to have a non-negative finite width (' + this._rectWidth + ')' );
      assert && assert( this._rectHeight >= 0 && isFinite( this._rectHeight ),
                                      'A rectangle needs to have a non-negative finite height (' + this._rectHeight + ')' );
      assert && assert( this._rectArcWidth >= 0 && isFinite( this._rectArcWidth ),
                                      'A rectangle needs to have a non-negative finite arcWidth (' + this._rectArcWidth + ')' );
      assert && assert( this._rectArcHeight >= 0 && isFinite( this._rectArcHeight ),
                                      'A rectangle needs to have a non-negative finite arcHeight (' + this._rectArcHeight + ')' );
      // assert && assert( !this.isRounded() || ( this._rectWidth >= this._rectArcWidth * 2 && this._rectHeight >= this._rectArcHeight * 2 ),
      //                                 'The rounded sections of the rectangle should not intersect (the length of the straight sections shouldn\'t be negative' );
      
      // sets our 'cache' to null, so we don't always have to recompute our shape
      this._shape = null;
      
      // should invalidate the path and ensure a redraw
      this.invalidateShape();
    },
    
    // accelerated hit detection for axis-aligned optionally-rounded rectangle
    // fast computation if it isn't rounded. if rounded, we check if a corner computation is needed (usually isn't), and only check that one needed corner
    containsPointSelf: function( point ) {
      var x = this._rectX;
      var y = this._rectY;
      var width = this._rectWidth;
      var height = this._rectHeight;
      var arcWidth = this._rectArcWidth;
      var arcHeight = this._rectArcHeight;
      var halfLine = this.getLineWidth() / 2;
      
      var result = true;
      if ( this._strokePickable ) {
        // test the outer boundary if we are stroke-pickable (if also fill-pickable, this is the only test we need)
        var rounded = this.isRounded();
        if ( !rounded && this.getLineJoin() === 'bevel' ) {
          // fall-back for bevel
          return Path.prototype.containsPointSelf.call( this, point );
        }
        var miter = this.getLineJoin() === 'miter' && !rounded;
        result = result && Rectangle.intersects( x - halfLine, y - halfLine,
                                                 width + 2 * halfLine, height + 2 * halfLine,
                                                 miter ? 0 : ( arcWidth + halfLine ), miter ? 0 : ( arcHeight + halfLine ),
                                                 point );
      }
      
      if ( this._fillPickable ) {
        if ( this._strokePickable ) {
          return result;
        } else {
          return Rectangle.intersects( x, y, width, height, arcWidth, arcHeight, point );
        }
      } else if ( this._strokePickable ) {
        return result && !Rectangle.intersects( x + halfLine, y + halfLine,
                                               width - 2 * halfLine, height - 2 * halfLine,
                                               arcWidth - halfLine, arcHeight - halfLine,
                                               point );
      } else {
        return false; // either fill nor stroke is pickable
      }
    },
    
    intersectsBoundsSelf: function( bounds ) {
      return !this.computeShapeBounds().intersection( bounds ).isEmpty();
    },
    
    // override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path
    paintCanvas: function( wrapper ) {
      var context = wrapper.context;
      
      // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
      if ( this.isRounded() ) {
        context.beginPath();
        var arcw = this._rectArcWidth;
        var arch = this._rectArcHeight;
        var lowX = this._rectX + arcw;
        var highX = this._rectX + this._rectWidth - arcw;
        var lowY = this._rectY + arch;
        var highY = this._rectY + this._rectHeight - arch;
        if ( arcw === arch ) {
          // we can use circular arcs, which have well defined stroked offsets
          context.arc( highX, lowY, arcw, -Math.PI / 2, 0, false );
          context.arc( highX, highY, arcw, 0, Math.PI / 2, false );
          context.arc( lowX, highY, arcw, Math.PI / 2, Math.PI, false );
          context.arc( lowX, lowY, arcw, Math.PI, Math.PI * 3 / 2, false );
        } else {
          // we have to resort to elliptical arcs
          context.ellipse( highX, lowY, arcw, arch, 0, -Math.PI / 2, 0, false );
          context.ellipse( highX, highY, arcw, arch, 0, 0, Math.PI / 2, false );
          context.ellipse( lowX, highY, arcw, arch, 0, Math.PI / 2, Math.PI, false );
          context.ellipse( lowX, lowY, arcw, arch, 0, Math.PI, Math.PI * 3 / 2, false );
        }
        context.closePath();
        
        if ( this._fill ) {
          this.beforeCanvasFill( wrapper ); // defined in Fillable
          context.fill();
          this.afterCanvasFill( wrapper ); // defined in Fillable
        }
        if ( this._stroke ) {
          this.beforeCanvasStroke( wrapper ); // defined in Strokable
          context.stroke();
          this.afterCanvasStroke( wrapper ); // defined in Strokable
        }
      } else {
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
      }
    },
    
    // create a rect instead of a path, hopefully it is faster in implementations
    createSVGFragment: function( svg, defs, group ) {
      return document.createElementNS( scenery.svgns, 'rect' );
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
    },
    
    setShape: function( shape ) {
      if ( shape !== null ) {
        throw new Error( 'Cannot set the shape of a scenery.Rectangle to something non-null' );
      } else {
        // probably called from the Path constructor
        this.invalidateShape();
      }
    },
    
    getShape: function() {
      if ( !this._shape ) {
        this._shape = this.createRectangleShape();
      }
      return this._shape;
    },
    
    hasShape: function() {
      return true;
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
      if ( this[privateName] !== value ) {
        this[privateName] = value;
        this.invalidateRectangle();
      }
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
  
  Rectangle.intersects = function( x, y, width, height, arcWidth, arcHeight, point ) {
    var result = point.x >= x &&
                 point.x <= x + width &&
                 point.y >= y &&
                 point.y <= y + height;
    
    if ( !result || arcWidth <= 0 || arcHeight <= 0 ) {
      return result;
    }
    
    // copy border-radius CSS behavior in Chrome, where the arcs won't intersect, in cases where the arc segments at full size would intersect each other
    var maximumArcSize = Math.min( width / 2, height / 2 );
    arcWidth = Math.min( maximumArcSize, arcWidth );
    arcHeight = Math.min( maximumArcSize, arcHeight );
    
    // we are rounded and inside the logical rectangle (if it didn't have rounded corners)
    
    // closest corner arc's center (we assume the rounded rectangle's arcs are 90 degrees fully, and don't intersect)
    var closestCornerX, closestCornerY, guaranteedInside = false;
    
    // if we are to the inside of the closest corner arc's center, we are guaranteed to be in the rounded rectangle (guaranteedInside)
    if ( point.x < x + width / 2 ) {
      closestCornerX = x + arcWidth;
      guaranteedInside = guaranteedInside || point.x >= closestCornerX;
    } else {
      closestCornerX = x + width - arcWidth;
      guaranteedInside = guaranteedInside || point.x <= closestCornerX;
    }
    if ( guaranteedInside ) { return true; }
    
    if ( point.y < y + height / 2 ) {
      closestCornerY = y + arcHeight;
      guaranteedInside = guaranteedInside || point.y >= closestCornerY;
    } else {
      closestCornerY = y + height - arcHeight;
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
    offsetX /= arcWidth;
    offsetY /= arcHeight;
    
    offsetX *= offsetX;
    offsetY *= offsetY;
    return offsetX + offsetY <= 1; // return whether we are in the rounded corner. see the formula for an ellipse
  };
  
  Rectangle.rect = function( x, y, width, height, options ) {
    return new Rectangle( x, y, width, height, 0, 0, options );
  };
  
  Rectangle.roundedRect = function( x, y, width, height, arcWidth, arcHeight, options ) {
    return new Rectangle( x, y, width, height, arcWidth, arcHeight, options );
  };
  
  Rectangle.bounds = function( bounds, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, 0, 0, options );
  };
  
  Rectangle.roundedBounds = function( bounds, arcWidth, arcHeight, options ) {
    return new Rectangle( bounds.minX, bounds.minY, bounds.width, bounds.height, arcWidth, arcHeight, options );
  };
  
  return Rectangle;
} );


