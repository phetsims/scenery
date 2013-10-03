// Copyright 2002-2013, University of Colorado

/**
 * Mix-in for nodes that support a standard fill.
 *
 * TODO: pattern and gradient handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var scenery = require( 'SCENERY/scenery' );
  
  scenery.Fillable = function Fillable( type ) {
    var proto = type.prototype;
    
    // this should be called in the constructor to initialize
    proto.initializeFillable = function() {
      this._fill = null;
      
      var that = this;
      this._fillListener = function() {
        that.invalidatePaint(); // TODO: move this to invalidateFill?
        that.invalidateFill();
      };
    };
    
    proto.hasFill = function() {
      return this._fill !== null;
    };
    
    proto.getFill = function() {
      return this._fill;
    };
    
    proto.setFill = function( fill ) {
      if ( this.getFill() !== fill ) {
        var hasInstances = this._instances.length > 0;
        
        if ( hasInstances && this._fill && this._fill.removeChangeListener ) {
          this._fill.removeChangeListener( this._fillListener );
        }
        
        this._fill = fill;
        
        if ( hasInstances && this._fill && this._fill.addChangeListener ) {
          this._fill.addChangeListener( this._fillListener );
        }
        
        this.invalidatePaint();
        
        this.invalidateFill();
      }
      return this;
    };
    
    var superFirstInstanceAdded = proto.firstInstanceAdded;
    proto.firstInstanceAdded = function() {
      if ( this._fill && this._fill.addChangeListener ) {
        this._fill.addChangeListener( this._fillListener );
      }
      
      if ( superFirstInstanceAdded ) {
        superFirstInstanceAdded.call( this );
      }
    };
    
    var superLastInstanceRemoved = proto.lastInstanceRemoved;
    proto.lastInstanceRemoved = function() {
      if ( this._fill && this._fill.removeChangeListener ) {
        this._fill.removeChangeListener( this._fillListener );
      }
      
      if ( superLastInstanceRemoved ) {
        superLastInstanceRemoved.call( this );
      }
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
      var style = 'fill: ';
      if ( !this._fill ) {
        // no fill
        style += 'none;';
      } else if ( this._fill.toCSS ) {
        // Color object fill
        style += this._fill.toCSS() + ';';
      } else if ( this._fill.getSVGDefinition ) {
        // reference the SVG definition with a URL
        style += 'url(#fill' + this.getId() + ');';
      } else {
        // plain CSS color
        style += this._fill + ';';
      }
      return style;
    };
    
    proto.isFillDOMCompatible = function() {
      // make sure we're not a pattern or gradient
      return !this._fill || !this._fill.getSVGDefinition;
    };
    
    proto.getCSSFill = function() {
      assert && assert( this.isFillDOMCompatible() );
      
      // if it's a Color object, get the corresponding CSS
      // 'transparent' will make us invisible if the fill is null
      return this._fill ? ( this._fill.toCSS ? this._fill.toCSS() : this._fill ) : 'transparent';
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
    
    proto.appendFillablePropString = function( spaces, result ) {
      if ( this._fill ) {
        if ( result ) {
          result += ',\n';
        }
        if ( typeof this._fill === 'string' ) {
          result += spaces + 'fill: \'' + this._fill + '\'';
        } else {
          result += spaces + 'fill: ' + this._fill.toString();
        }
      }
      
      return result;
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


