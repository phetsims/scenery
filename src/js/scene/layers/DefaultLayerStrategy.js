// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  
  // specified as such, since there is no needed shared state (we can have node.layerStrategy = scenery.DefaultLayerStrategy for many nodes)
  scenery.DefaultLayerStrategy = {
    enter: function( node, layerState ) {
      
      // if the node isn't self-rendering, we can skip it completely
      if ( node.hasSelf() ) {
        // check if we need to change layer types
        if ( !layerState.getCurrentLayerType() || !layerState.getCurrentLayerType().supportsNode( node ) ) {
          var supportedTypes = node._supportedLayerTypes;
          
          var preferredType = layerState.bestPreferredLayerTypeFor( supportedTypes );
          if ( preferredType ) {
            layerState.switchToType( preferredType );
          } else {
            layerState.switchToType( supportedTypes[0] );
          }
        }
        
        // trigger actual layer creation if necessary (allow collapsing of layers otherwise)
        layerState.markSelf();
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
  };
  
})();
