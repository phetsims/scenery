// Copyright 2002-2012, University of Colorado

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  // static references needed for type initialization
  var CanvasLayer = require( 'SCENERY/layers/CanvasLayer' );
  var DOMLayer = require( 'SCENERY/layers/DOMLayer' );
  var SVGLayer = require( 'SCENERY/layers/SVGLayer' );
  
  scenery.LayerType = function( Constructor, name ) {
    this.Constructor = Constructor;
    this.name = name;
  };
  var LayerType = scenery.LayerType;
  
  LayerType.prototype = {
    constructor: LayerType,
    
    // able to override this for layer types that support the features but extend the abilities
    supports: function( type ) {
      return this === type;
    },
    
    supportsNode: function( node ) {
      var that = this;
      return _.some( node._supportedLayerTypes, function( layerType ) {
        return that.supports( layerType );
      } );
    },
    
    createLayer: function( args ) {
      var Constructor = this.Constructor;
      return new Constructor( args );
    }
  };
  
  LayerType.Canvas = new LayerType( CanvasLayer, 'canvas' );
  LayerType.DOM = new LayerType( DOMLayer, 'dom' );
  LayerType.SVG = new LayerType( SVGLayer, 'svg' );
  
  return LayerType;
} );


