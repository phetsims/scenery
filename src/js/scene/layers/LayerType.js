// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
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
  
  LayerType.Canvas = new scenery.LayerType( scenery.CanvasLayer, 'canvas' );
  LayerType.DOM = new scenery.LayerType( scenery.DOMLayer, 'dom' );
  LayerType.SVG = new scenery.LayerType( scenery.SVGLayer, 'svg' );
  
})();


