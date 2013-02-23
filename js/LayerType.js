// Copyright 2002-2012, University of Colorado

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var CanvasLayer = require( 'SCENERY/CanvasLayer' );
  var DOMLayer = require( 'SCENERY/DOMLayer' );
  var SVGLayer = require( 'SCENERY/SVGLayer' );
  
  var LayerType = function( Constructor, name ) {
    this.Constructor = Constructor;
    this.name = name;
  };
  
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


