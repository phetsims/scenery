// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  scenery.LayerState = function() {
    this.preferredLayerTypes = [];
    
    this.typeDirty = true;
    this.nextLayerType = null;
    
    /*
      TODO:
      
      compact layers as necessary
      hook them up as a linked list
    */
  }
  
  var LayerState = scenery.LayerState;
  LayerState.prototype = {
    constructor: LayerState,
    
    pushPreferredLayerType: function( layerType ) {
      this.preferredLayerTypes.push( layerType );
    },
    
    popPreferredLayerType: function( layerType ) {
      this.preferredLayerTypes.pop();
    },
    
    getPreferredLayerType: function() {
      if ( this.preferredLayerTypes.length !== 0 ) {
        return this.preferredLayerTypes[this.preferredLayerTypes.length - 1];
      } else {
        return null;
      }
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
    },
    
    bestPreferredLayerTypeFor: function( defaultTypeOptions ) {
      for ( var i = this.preferredLayerTypes.length - 1; i >= 0; i-- ) {
        var preferredType = this.preferredLayerTypes[i];
        if ( _.some( defaultTypeOptions, function( defaultType ) { return preferredType.supports( defaultType ); } ) ) {
          return preferredType;
        }
      }
      
      // none of our stored preferred layer types are able to support any of the default type options
      return null;
    }
  };
})();
