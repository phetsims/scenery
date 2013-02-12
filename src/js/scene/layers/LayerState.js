// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  scenery.LayerState = function() {
    this.layers = [];
    this.lastLayer = null;
    
    /*
      TODO:
      
      compact layers as necessary
      hook them up as a linked list
    
      - compacts layers as necessary (don't instantiate until something hasSelf())
      push/pop preferredLayerType
      switchToType()
      self() -- when node hasSelf(), ensure that we finalize a layer switch
      query current layer type
      other queries
    */
  }
  
  var LayerState = scenery.LayerState;
  LayerState.prototype = {
    constructor: LayerState,
    
    pushPreferredLayerType: function( layerType ) {
      // TODO
    },
    
    popPreferredLayerType: function( layerType ) {
      // TODO
    },
    
    getPreferredLayerType: function() {
      // TODO
    },
    
    switchToType: function( layerType ) {
      // TODO
    },
    
    // called so that we can finalize a layer switch (instead of collapsing unneeded layers)
    markSelf: function() {
      // TODO
    },
    
    getCurrentLayerType: function() {
      // TODO
    }
  };
})();
