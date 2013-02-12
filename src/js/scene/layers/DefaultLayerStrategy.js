// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  
  // specified as such, since there is no needed shared state (we can have node.layerStrategy = scenery.DefaultLayerStrategy for many nodes)
  scenery.DefaultLayerStrategy = {
    enter: function( node, layerState ) {
      // check if we need to change layer types
      if ( node.hasSelf() && !layerState.getCurrentLayerType().supportsNode( node ) ) {
        
        var preferredType = layerState.getPreferredLayerType();
        if ( preferredType && preferredType.supportsNode( node ) ) {
          layerState.switchToType( preferredType );
        } else {
          layerState.switchToType( node._supportedLayerTypes[0] );
        }
      }
    },
    
    afterSelf: function( node, layerState ) {
      // no-op, and possibly not used
    },
    
    betweenChildren: function( node, layerState ) {
      // no-op, and possibly not used
    },
    
    exit: function( node, layerState ) {
      // currently a no-op
    }
  }
  
})();
