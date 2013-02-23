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
    
    // on mutation, set the fill parameter first
    proto._mutatorKeys = [ 'fill' ].concat( proto._mutatorKeys );
    
    Object.defineProperty( proto, 'fill', { set: proto.setFill, get: proto.getFill } );
    
    if ( !proto.invalidateFill ) {
      proto.invalidateFill = function() {
        // override if fill handling is necessary (TODO: mixins!)
      };
    }
  };
} );


