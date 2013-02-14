// Copyright 2002-2012, University of Colorado

var scenery = scenery || {};

(function(){
  
  // specified as such, since there is no needed shared state (we can have node.layerStrategy = scenery.DefaultLayerStrategy for many nodes)
  scenery.DefaultLayerStrategy = {
    enter: function( trail, layerState ) {
      var node = trail.lastNode();
      
      // if the node isn't self-rendering, we can skip it completely
      if ( node.hasSelf() ) {
        // check if we need to change layer types
        if ( !layerState.getCurrentLayerType() || !layerState.getCurrentLayerType().supportsNode( node ) ) {
          var supportedTypes = node._supportedLayerTypes;
          
          var preferredType = layerState.bestPreferredLayerTypeFor( supportedTypes );
          if ( preferredType ) {
            layerState.switchToType( trail, preferredType );
          } else {
            layerState.switchToType( trail, supportedTypes[0] );
          }
        }
        
        // trigger actual layer creation if necessary (allow collapsing of layers otherwise)
        layerState.markSelf( trail );
      }
    },
    
    afterSelf: function( trail, layerState ) {
      // no-op, and possibly not used
    },
    
    betweenChildren: function( trail, layerState ) {
      // no-op, and possibly not used
    },
    
    exit: function( trail, layerState ) {
      // currently a no-op
    }
  };
  
})();
