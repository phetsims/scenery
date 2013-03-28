// Copyright 2002-2012, University of Colorado

/**
 * Mix-in for nodes that support a standard stroke.
 *
 * TODO: miterLimit handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  var LineStyles = require( 'KITE/util/LineStyles' );
  
  scenery.Strokable = function( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeStrokable = function() {
      this._stroke = null;
      this._lineDrawingStyles = new LineStyles();
    };
    
    proto.hasStroke = function() {
      return this._stroke !== null;
    };
    
    proto.getLineWidth = function() {
      return this._lineDrawingStyles.lineWidth;
    };
    
    proto.setLineWidth = function( lineWidth ) {
      if ( this.getLineWidth() !== lineWidth ) {
        this.markOldSelfPaint(); // since the previous line width may have been wider
        
        this._lineDrawingStyles.lineWidth = lineWidth;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineCap = function() {
      return this._lineDrawingStyles.lineCap;
    };
    
    proto.setLineCap = function( lineCap ) {
      if ( this._lineDrawingStyles.lineCap !== lineCap ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineCap = lineCap;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineJoin = function() {
      return this._lineDrawingStyles.lineJoin;
    };
    
    proto.setLineJoin = function( lineJoin ) {
      if ( this._lineDrawingStyles.lineJoin !== lineJoin ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineJoin = lineJoin;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.getLineDash = function() {
      return this._lineDrawingStyles.lineDash;
    };
    
    proto.setLineDash = function( lineDash ) {
      if ( this._lineDrawingStyles.lineDash !== lineDash ) {
        this.markOldSelfPaint();
        
        this._lineDrawingStyles.lineDash = lineDash;
        
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.setLineStyles = function( lineStyles ) {
      // TODO: since we have been using lineStyles as mutable for now, lack of change check is good here?
      this.markOldSelfPaint();
      
      this._lineDrawingStyles = lineStyles;
      this.invalidateStroke();
      return this;
    };
    
    proto.getLineStyles = function() {
      return this._lineDrawingStyles;
    };
    
    proto.getStroke = function() {
      return this._stroke;
    };
    
    proto.setStroke = function( stroke ) {
      if ( this.getStroke() !== stroke ) {
        // since this can actually change the bounds, we need to handle a few things differently than the fill
        this.markOldSelfPaint();
        
        this._stroke = stroke;
        this.invalidateStroke();
      }
      return this;
    };
    
    proto.beforeCanvasStroke = function( layer ) {
      // TODO: is there a better way of not calling so many things on each stroke?
      layer.setStrokeStyle( this._stroke );
      layer.setLineWidth( this.getLineWidth() );
      layer.setLineCap( this.getLineCap() );
      layer.setLineJoin( this.getLineJoin() );
      layer.setLineDash( this.getLineDash() );
      if ( this._stroke.transformMatrix ) {
        layer.context.save();
        this._stroke.transformMatrix.canvasAppendTransform( layer.context );
      }
    };
    
    proto.afterCanvasStroke = function( layer ) {
      if ( this._stroke.transformMatrix ) {
        layer.context.restore();
      }
    };
    
    proto.getSVGStrokeStyle = function() {
      // if the style has an SVG definition, use that with a URL reference to it
      var style = 'stroke: ' + ( this._stroke ? ( this._stroke.getSVGDefinition ? 'url(#stroke' + this.getId() + ')' : this._stroke ) : 'none' ) + ';';
      if ( this._stroke ) {
        // TODO: don't include unnecessary directives?
        style += 'stroke-width: ' + this.getLineWidth() + ';';
        style += 'stroke-linecap: ' + this.getLineCap() + ';';
        style += 'stroke-linejoin: ' + this.getLineJoin() + ';';
        if ( this.getLineDash() ) {
          style += 'stroke-dasharray: ' + this.getLineDash().join( ',' ) + ';';
        }
      }
      return style;
    };
    
    proto.addSVGStrokeDef = function( svg, defs ) {
      var stroke = this.getStroke();
      var strokeId = 'stroke' + this.getId();
      
      // add new definitions if necessary
      if ( stroke && stroke.getSVGDefinition ) {
        defs.appendChild( stroke.getSVGDefinition( strokeId ) );
      }
    };
    
    proto.removeSVGStrokeDef = function( svg, defs ) {
      var strokeId = 'stroke' + this.getId();
      
      // wipe away any old definition
      var oldStrokeDef = svg.getElementById( strokeId );
      if ( oldStrokeDef ) {
        defs.removeChild( oldStrokeDef );
      }
    };
    
    // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
    proto._mutatorKeys = [ 'stroke', 'lineWidth', 'lineCap', 'lineJoin', 'lineDash' ].concat( proto._mutatorKeys );
    
    // TODO: miterLimit support?
    Object.defineProperty( proto, 'stroke', { set: proto.setStroke, get: proto.getStroke } );
    Object.defineProperty( proto, 'lineWidth', { set: proto.setLineWidth, get: proto.getLineWidth } );
    Object.defineProperty( proto, 'lineCap', { set: proto.setLineCap, get: proto.getLineCap } );
    Object.defineProperty( proto, 'lineJoin', { set: proto.setLineJoin, get: proto.getLineJoin } );
    Object.defineProperty( proto, 'lineDash', { set: proto.setLineDash, get: proto.getLineDash } );
    
    if ( !proto.invalidateStroke ) {
      proto.invalidateStroke = function() {
        // override if stroke handling is necessary (TODO: mixins!)
      };
    }
  };
  var Strokable = scenery.Strokable;
  
  return Strokable;
} );


