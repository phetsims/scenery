// Copyright 2002-2012, University of Colorado

/**
 * A rectangular node that inherits Path, and allows for optimized drawing,
 * and improved rectangle handling.
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
  
  inherit( Rectangle, Path, {
    isRounded: function() {
      return this._rectArcWidth !== 0 && this._rectArcHeight !== 0;
    },
    
    createRectangleShape: function() {
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
    
    // override paintCanvas with a faster version, since fillRect and drawRect don't affect the current default path
    paintCanvas: function( state ) {
      // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
      if ( this.isRounded() ) {
        return Path.prototype.paintCanvas.call( this, state );
      }
      
      var layer = state.layer;
      var context = layer.context;

      // TODO: how to handle fill/stroke delay optimizations here?
      if ( this._fill ) {
        this.beforeCanvasFill( layer ); // defined in Fillable
        context.fillRect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
        this.afterCanvasFill( layer ); // defined in Fillable
      }
      if ( this._stroke ) {
        this.beforeCanvasStroke( layer ); // defined in Strokable
        context.strokeRect( this._rectX, this._rectY, this._rectWidth, this._rectHeight );
        this.afterCanvasStroke( layer ); // defined in Strokable
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


