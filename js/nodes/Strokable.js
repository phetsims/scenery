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
  var LineStyles = require( 'SCENERY/shapes/util/LineStyles' );
  
  require( 'SCENERY/shapes/Shape' );
  
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
    
    // on mutation, set the stroke parameters first since they may affect the bounds (and thus later operations)
    proto._mutatorKeys = [ 'stroke', 'lineWidth', 'lineCap', 'lineJoin' ].concat( proto._mutatorKeys );
    
    Object.defineProperty( proto, 'stroke', { set: proto.setStroke, get: proto.getStroke } );
    Object.defineProperty( proto, 'lineWidth', { set: proto.setLineWidth, get: proto.getLineWidth } );
    Object.defineProperty( proto, 'lineCap', { set: proto.setLineCap, get: proto.getLineCap } );
    Object.defineProperty( proto, 'lineJoin', { set: proto.setLineJoin, get: proto.getLineJoin } );
    
    if ( !proto.invalidateStroke ) {
      proto.invalidateStroke = function() {
        // override if stroke handling is necessary (TODO: mixins!)
      };
    }
  };
  var Strokable = scenery.Strokable;
  
  return Strokable;
} );


