// Copyright 2002-2012, University of Colorado

define( function( require ) {
  "use strict";
  
  // TODO: remove rats-nest of mutable shared state that we have here, or clearly doc what are transient instances vs immutable
  scenery.LayerState = function() {
    this.preferredLayerTypes = [];
    
    this.resetInternalState();
  };
  
  var LayerState = scenery.LayerState;
  LayerState.prototype = {
    constructor: LayerState,
    
    /*
     * Construct a list of layer entries between two Trails (inclusive).
     * Each element of the returned array will have { type: <layer type>, start: <start trail>, end: <end trail> }
     * startingLayerType can be null to signify there is no preceeding layer
     */
    buildLayers: function( startPointer, endPointer, startingLayerType ) {
      // TODO: accept initial layer in args?
      this.resetInternalState();
      
      this.currentLayerStartPointer = startPointer;
      
      if ( startingLayerType ) {
        this.nextLayerType = startingLayerType;
      }
      
      var state = this;
      
      startPointer.depthFirstUntil( endPointer, function( pointer ) {
        state.currentPointer = pointer;
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          node.layerStrategy.enter( pointer.trail, state );
        } else {
          node.layerStrategy.exit( pointer.trail, state );
        }
      }, false ); // don't exclude endpoints
      
      this.currentPointer = endPointer;
      this.finishLayer( endPointer );
      
      return this.layerChangeEntries;
    },
    
    resetInternalState: function() {
      this.layerChangeEntries = [];
      this.typeDirty = true;
      this.nextLayerType = null;
      
      this.currentLayerStartPointer = null;
      this.lastSelfTrail = null;
    },
    
    pushPreferredLayerType: function( layerType ) {
      this.preferredLayerTypes.push( layerType );
    },
    
    popPreferredLayerType: function() {
      this.preferredLayerTypes.pop();
    },
    
    switchToType: function( trail, layerType ) {
      var isStart = this.nextLayerType === null;
      this.typeDirty = true;
      this.nextLayerType = layerType;
      if ( !isStart ) {
        this.currentLayerStartPointer = this.currentPointer.copy();
      }
    },
    
    // called so that we can finalize a layer switch (instead of collapsing unneeded layers)
    markSelf: function() {
      var trail = this.currentPointer.trail;
      
      if ( this.typeDirty ) {
        this.layerChange( trail );
      }
      this.lastSelfTrail = trail.copy();
    },
    
    // can be null to indicate that there is no current layer type
    getCurrentLayerType: function() {
      return this.nextLayerType;
    },
    
    finishLayer: function( endPointer ) {
      if ( this.layerChangeEntries.length ) {
        var entry = this.layerChangeEntries[this.layerChangeEntries.length-1];
        entry.endSelfTrail = this.lastSelfTrail;
        entry.endPointer = endPointer;
      }
    },
    
    layerChange: function( firstSelfTrail ) {
      this.typeDirty = false;
      
      var previousPointer = this.currentLayerStartPointer.copy();
      previousPointer.nestedBackwards();
      this.finishLayer( previousPointer );
      
      this.layerChangeEntries.push( {
        type: this.nextLayerType,
        startPointer: this.currentLayerStartPointer.copy(),
        startSelfTrail: firstSelfTrail,
      } );
    },
    
    getPreferredLayerType: function() {
      if ( this.preferredLayerTypes.length !== 0 ) {
        return this.preferredLayerTypes[this.preferredLayerTypes.length - 1];
      } else {
        return null;
      }
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
} );
