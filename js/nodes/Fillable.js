// Copyright 2002-2012, University of Colorado

/**
 * Mix-in for nodes that support a standard fill.
 *
 * TODO: pattern and gradient handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Fillable = function( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeFillable = function() {
      this._fill = null;
    };
    
    proto.hasFill = function() {
      return this._fill !== null;
    };
    
    proto.getFill = function() {
      return this._fill;
    };
    
    proto.setFill = function( fill ) {
      if ( this.getFill() !== fill ) {
        this._fill = fill;
        this.invalidatePaint();
        
        this.invalidateFill();
      }
      return this;
    };
    
    proto.beforeCanvasFill = function( wrapper ) {
      wrapper.setFillStyle( this._fill );
      if ( this._fill.transformMatrix ) {
        wrapper.context.save();
        this._fill.transformMatrix.canvasAppendTransform( wrapper.context );
      }
    };
    
    proto.afterCanvasFill = function( wrapper ) {
      if ( this._fill.transformMatrix ) {
        wrapper.context.restore();
      }
    };
    
    proto.getSVGFillStyle = function() {
      // if the fill has an SVG definition, use that with a URL reference to it
      return 'fill: ' + ( this._fill ? ( this._fill.getSVGDefinition ? 'url(#fill' + this.getId() + ')' : this._fill ) : 'none' ) + ';';
    };
    
    proto.addSVGFillDef = function( svg, defs ) {
      var fill = this.getFill();
      var fillId = 'fill' + this.getId();
      
      // add new definitions if necessary
      if ( fill && fill.getSVGDefinition ) {
        defs.appendChild( fill.getSVGDefinition( fillId ) );
      }
    };
    
    proto.removeSVGFillDef = function( svg, defs ) {
      var fillId = 'fill' + this.getId();
      
      // wipe away any old definition
      var oldFillDef = svg.getElementById( fillId );
      if ( oldFillDef ) {
        defs.removeChild( oldFillDef );
      }
    };
    
    // on mutation, set the fill parameter first
    proto._mutatorKeys = [ 'fill' ].concat( proto._mutatorKeys );
    
    Object.defineProperty( proto, 'fill', { set: proto.setFill, get: proto.getFill } );
    
    if ( !proto.invalidateFill ) {
      proto.invalidateFill = function() {
        // override if fill handling is necessary (TODO: mixins!)
      };
    }
  };
  var Fillable = scenery.Fillable;
  
  return Fillable;
} );


