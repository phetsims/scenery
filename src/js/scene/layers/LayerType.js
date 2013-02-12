// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.LayerType = function( Constructor ) {
    this.Constructor = Constructor;
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
        that.supports( layerType );
      } );
    },
    
    createLayer: function( args ) {
      var Constructor = this.Constructor;
      return new Constructor( args );
    }
  };
  
  LayerType.Canvas = new scenery.LayerType( scenery.CanvasLayer );
  LayerType.DOM = new scenery.LayerType( scenery.DOMLayer );
  
})();


