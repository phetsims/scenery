// Copyright 2002-2012, University of Colorado

/**
 * A layer state is used to construct layer information (and later, layers), and is a state machine
 * that layer strategies from each node modify. Iterating through all of the nodes in a depth-first
 * manner will modify the LayerBuilder so that layer information can be retrieved.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/layers/LayerBoundary' );
  require( 'SCENERY/Trail' );
  require( 'SCENERY/TrailPointer' );
  
  /*
   * Builds layer information between trails
   *
   * previousLayerType should be null if there is no previous layer.
   */
  scenery.LayerBuilder = function( scene, previousLayerType, previousSelfTrail, nextSelfTrail ) {
    var builder = this;
    
    /*---------------------------------------------------------------------------*
    * Initial state
    *----------------------------------------------------------------------------*/
    
    this.layerTypeStack = [];
    this.boundaries = [];
    this.pendingBoundary = new scenery.LayerBoundary(); // TODO: ensure all of the properties are set for the 'previous'
    this.pendingBoundary.previousLayerType = previousLayerType;
    this.pendingBoundary.previousSelfTrail = previousSelfTrail;
    
    /*
     * The current layer type active, and whether it has been 'used' yet. A node with hasSelf() will trigger a 'used' action,
     * and if the layer hasn't been used, it will actually trigger a boundary creation. We want to collapse 'unused' layers
     * and boundaries together, so that every created layer has a node that displays something.
     */
    this.currentLayerType = previousLayerType;
    this.layerChangePending = previousSelfTrail === null;
    
    /*---------------------------------------------------------------------------*
    * Iteration
    *----------------------------------------------------------------------------*/
    
    var startPointer;
    if ( previousSelfTrail ) {
      // Move our start pointer just past the previousSelfTrail, since our previousLayerType is presumably for that trail's node's self.
      // Anything after that self could have been collapsed, so we need to start there.
      startPointer = new scenery.TrailPointer( previousSelfTrail.copy(), true );
      startPointer.nestedForwards();
    } else {
      startPointer = new scenery.TrailPointer( new scenery.Trail( scene ), true );
    }
    
    var endPointer;
    if ( nextSelfTrail ) {
      // include the nextSelfTrail's 'before' in our iteration, so we can stitch properly with the next layer
      endPointer = new scenery.TrailPointer( nextSelfTrail.copy(), true );
    } else {
      endPointer = new scenery.TrailPointer( new scenery.Trail( scene ), false );
    }
    
    startPointer.depthFirstUntil( endPointer, function( pointer ) {
      var node = pointer.trail.lastNode();
      
      if ( pointer.isBefore ) {
        node.layerStrategy.enter( pointer, builder );
      } else {
        node.layerStrategy.exit( pointer, builder );
      }
    }, false ); // include the endpoints
    
    // special case handling if we are at the 'end' of the scene, so that we create another 'wrapping' boundary
    if ( nextSelfTrail === null ) {
      this.pendingBoundary.previousEndPointer = endPointer; // TODO: consider implications if we leave this null, to indicate that it is not ended?
      this.layerChange( null );
    }
    
    /*
     * LayerBoundary properties and assurances:
     *
     * previousLayerType  - initialized in constructor (in case there are no layer changes)
     *                      set in layerChange for "fresh" pending boundary
     * nextLayerType      - set and overwrites in switchToType, for collapsing layers
     *                      not set anywhere else, so we can leave it null
     * previousSelfTrail  - initialized in constructor
     *                      updated in markSelf if there is no pending change (don't set if there is a pending change)
     * nextSelfTrail      - set on layerChange for "stale" boundary
     *                      stays null if nextSelfTrail === null
     * previousEndPointer - (normal boundary) set in switchToType if there is no layer change pending
     *                      set in finalization if nextSelfTrail === null && !this.layerChangePending (previousEndPointer should be null in that case)
     * nextStartPointer   - set in switchToType (always), overwrites values so we collapse layers nicely
     */
  };
  var LayerBuilder = scenery.LayerBuilder;
  
  LayerBuilder.prototype = {
    constructor: LayerBuilder,
    
    switchToType: function( pointer, layerType ) {
      this.currentLayerType = layerType;
      
      this.pendingBoundary.nextLayerType = layerType;
      this.pendingBoundary.nextStartPointer = pointer.copy();
      if ( !this.layerChangePending ) {
        this.pendingBoundary.previousEndPointer = pointer.copy();
      }
      
      this.layerChangePending = true; // we wait until the first markSelf() call to create a boundary
    },
    
    // called so that we can finalize a layer switch (instead of collapsing unneeded layers)
    markSelf: function( pointer ) {
      if ( this.layerChangePending ) {
        this.layerChange( pointer );
      } else {
        // TODO: performance-wise, don't lookup indices on this copy? make a way to create a lightweight copy?
        this.pendingBoundary.previousSelfTrail = pointer.trail.copy();
      }
    },
    
    // can be null to indicate that there is no current layer type
    getCurrentLayerType: function() {
      return this.currentLayerType;
    },
    
    // allows selfPointer === null at the end if the main iteration's nextSelfTrail === null (i.e. we are at the end of the scene)
    layerChange: function( selfPointer ) {
      this.layerChangePending = false;
      
      var confirmedBoundary = this.pendingBoundary;
      
      confirmedBoundary.nextSelfTrail = selfPointer ? selfPointer.trail : null;
      
      this.boundaries.push( confirmedBoundary );
      
      this.pendingBoundary = new scenery.LayerBoundary();
      this.pendingBoundary.previousLayerType = confirmedBoundary.nextLayerType;
      this.pendingBoundary.previousSelfTrail = confirmedBoundary.nextSelfTrail;
    },
    
    pushPreferredLayerType: function( layerType ) {
      this.layerTypeStack.push( layerType );
    },
    
    popPreferredLayerType: function() {
      this.layerTypeStack.pop();
    },
    
    getPreferredLayerType: function() {
      if ( this.layerTypeStack.length !== 0 ) {
        return this.layerTypeStack[this.layerTypeStack.length - 1];
      } else {
        return null;
      }
    },
    
    bestPreferredLayerTypeFor: function( renderers ) {
      for ( var i = this.layerTypeStack.length - 1; i >= 0; i-- ) {
        var preferredType = this.layerTypeStack[i];
        if ( _.some( renderers, function( renderer ) { return preferredType.supportsRenderer( renderer ); } ) ) {
          return preferredType;
        }
      }
      
      // none of our stored preferred layer types are able to support any of the default type options
      return null;
    }
  };
  
  return LayerBuilder;
} );
