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
      sceneryAssert && sceneryAssert( this._rectWidth >= 0 && isFinite( this._rectWidth ), 'A rectangle needs to have a non-negative finite width (' + this._rectWidth + ')' );
      sceneryAssert && sceneryAssert( this._rectHeight >= 0 && isFinite( this._rectHeight ), 'A rectangle needs to have a non-negative finite height (' + this._rectHeight + ')' );
      sceneryAssert && sceneryAssert( this._rectArcWidth >= 0 && isFinite( this._rectArcWidth ), 'A rectangle needs to have a non-negative finite arcWidth (' + this._rectArcWidth + ')' );
      sceneryAssert && sceneryAssert( this._rectArcHeight >= 0 && isFinite( this._rectArcHeight ), 'A rectangle needs to have a non-negative finite arcHeight (' + this._rectArcHeight + ')' );
      
      if ( this.isRounded() ) {
        return Shape.roundRectangle( this._rectX, this._rectY, this._rectWidth, this._rectHeight, this._rectArcWidth, this._rectArcHeight );
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


