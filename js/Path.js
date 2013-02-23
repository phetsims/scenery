// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * TODO: newlines
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/Node' );
  var LayerType = require( 'SCENERY/LayerType' );
  var fillable = require( 'SCENERY/Fillable' );
  var strokable = require( 'SCENERY/Strokable' );
  var objectCreate = require( 'SCENERY/Util' ).objectCreate;
  
  scenery.Path = function( options ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    this._shape = null;
    
    // ensure we have a parameter object
    options = options || {};
    
    this.initializeStrokable();
    
    Node.call( this, options );
  };
  var Path = scenery.Path;
  
  Path.prototype = objectCreate( Node.prototype );
  Path.prototype.constructor = Path;
  
  // sets the shape drawn, or null to remove the shape
  Path.prototype.setShape = function( shape ) {
    if ( this._shape !== shape ) {
      this._shape = shape;
      this.invalidateShape();
    }
    return this;
  };
  
  Path.prototype.getShape = function() {
    return this._shape;
  };
  
  Path.prototype.invalidateShape = function() {
    this.markOldSelfPaint();
    
    if ( this.hasShape() ) {
      this.invalidateSelf( this._shape.computeBounds( this._stroke ? this._lineDrawingStyles : null ) );
      this.invalidatePaint();
    }
  };
  
  // hook stroke mixin changes to invalidation
  Path.prototype.invalidateStroke = function() {
    this.invalidateShape();
  };
  
  Path.prototype.hasShape = function() {
    return this._shape !== null;
  };
  
  Path.prototype.paintCanvas = function( state ) {
    if ( this.hasShape() ) {
      var layer = state.layer;
      var context = layer.context;

      // TODO: fill/stroke delay optimizations?
      context.beginPath();
      this._shape.writeToContext( context );

      if ( this._fill ) {
        layer.setFillStyle( this._fill );
        context.fill();
      }
      if ( this._stroke ) {
        layer.setStrokeStyle( this._stroke );
        layer.setLineWidth( this.getLineWidth() );
        layer.setLineCap( this.getLineCap() );
        layer.setLineJoin( this.getLineJoin() );
        context.stroke();
      }
    }
  };
  
  Path.prototype.paintWebGL = function( state ) {
    throw new Error( 'Path.prototype.paintWebGL unimplemented' );
  };
  
  Path.prototype.createSVGFragment = function() {
    var path = document.createElementNS( 'http://www.w3.org/2000/svg', 'path' );
    this.updateSVGFragment( path );
    return path;
  };
  
  Path.prototype.updateSVGFragment = function( path ) {
    if ( this.hasShape() ) {
      path.setAttribute( 'd', this._shape.getSVGPath() );
    } else if ( path.hasAttribute( 'd' ) ) {
      path.removeAttribute( 'd' );
    }
    
    var style = '';
    style += 'fill: ' + ( this._fill ? this._fill : 'none' ) + ';'; // TODO: handling patterns and gradients!
    style += 'stroke: ' + ( this._stroke ? this._stroke : 'none' ) + ';';
    if ( this._stroke ) {
      // TODO: don't include unnecessary directives?
      style += 'stroke-width: ' + this.getLineWidth() + ';';
      style += 'stroke-linecap: ' + this.getLineCap() + ';';
      style += 'stroke-linejoin: ' + this.getLineJoin() + ';';
    }
    path.setAttribute( 'style', style );
  };
  
  Path.prototype.hasSelf = function() {
    return true;
  };
  
  // override for computation of whether a point is inside the self content
  // point is considered to be in the local coordinate frame
  Path.prototype.containsPointSelf = function( point ) {
    if ( !this.hasShape() ) {
      return false;
    }
    
    var result = this._shape.containsPoint( point );
    
    // also include the stroked region in the hit area if applicable
    if ( !result && this._includeStrokeInHitRegion && this.hasStroke() ) {
      result = this._shape.getStrokedShape( this._lineDrawingStyles ).containsPoint( point );
    }
    return result;
  };
  
  // whether this node's self intersects the specified bounds, in the local coordinate frame
  Path.prototype.intersectsBoundsSelf = function( bounds ) {
    // TODO: should a shape's stroke be included?
    return this.hasShape() ? this._shape.intersectsBounds( bounds ) : false;
  };
  
  // TODO: stroke / fill mixins
  Path.prototype._mutatorKeys = [ 'shape' ].concat( Node.prototype._mutatorKeys );
  
  Path.prototype._supportedLayerTypes = [ LayerType.Canvas, LayerType.SVG ];
  
  Object.defineProperty( Path.prototype, 'shape', { set: Path.prototype.setShape, get: Path.prototype.getShape } );
  
  // mix in fill/stroke handling code. for now, this is done after 'shape' is added to the mutatorKeys so that stroke parameters
  // get set first
  fillable( Path );
  strokable( Path );
  
  return Path;
} );


