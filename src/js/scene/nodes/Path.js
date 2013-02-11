// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * TODO: newlines
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.Path = function( params ) {
    // TODO: consider directly passing in a shape object (or at least handling that case)
    this._shape = null;
    
    // ensure we have a parameter object
    params = params || {};
    
    this.initializeStrokable();
    
    scenery.Node.call( this, params );
  };
  var Path = scenery.Path;
  
  Path.prototype = phet.Object.create( scenery.Node.prototype );
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
  
  Path.prototype.renderSelf = function( state ) {
    if ( this.hasShape() ) {
      if ( state.isCanvasState() ) {
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
      } else {
        throw new Error( 'layer type shape rendering not implemented' );
      }
    }
  };
  
  // override for computation of whether a point is inside the content rendered in renderSelf
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
  Path.prototype._mutatorKeys = [ 'shape' ].concat( scenery.Node.prototype._mutatorKeys );
  
  Object.defineProperty( Path.prototype, 'shape', { set: Path.prototype.setShape, get: Path.prototype.getShape } );
  
  // mix in stroke handling code. for now, this is done after 'shape' is added to the mutatorKeys so that stroke parameters
  // get set first
  scenery.Strokable( Path );
  
})();


