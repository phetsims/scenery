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
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  
  /*
   * Builds layer information between trails
   *
   * previousLayerType should be null if there is no previous layer.
   */
  scenery.LayerBuilder = function( scene, previousLayerType, previousSelfTrail, nextSelfTrail ) {
    
    /*---------------------------------------------------------------------------*
    * Initial state
    *----------------------------------------------------------------------------*/
    
    this.layerTypeStack = [];
    this.boundaries = [];
    this.pendingBoundary = new scenery.LayerBoundary();
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
    * Start / End pointers
    *----------------------------------------------------------------------------*/
    
    if ( previousSelfTrail ) {
      // Move our start pointer just past the previousSelfTrail, since our previousLayerType is presumably for that trail's node's self.
      // Anything after that self could have been collapsed, so we need to start there.
      this.startPointer = new scenery.TrailPointer( previousSelfTrail.copy(), true );
      this.startPointer.nestedForwards();
    } else {
      this.startPointer = new scenery.TrailPointer( new scenery.Trail( scene ), true );
    }
    
    if ( nextSelfTrail ) {
      // include the nextSelfTrail's 'before' in our iteration, so we can stitch properly with the next layer
      this.endPointer = new scenery.TrailPointer( nextSelfTrail.copy(), true );
    } else {
      this.endPointer = new scenery.TrailPointer( new scenery.Trail( scene ), false );
    }
    
    this.includesEndTrail = nextSelfTrail !== null;
    
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
    
    // walks part of the state up to just before the startPointer. we want the preferred layer stack to be in place, but the rest is not important
    prepareLayerStack: function() {
      var pointer = new scenery.TrailPointer( new scenery.Trail( this.startPointer.trail.rootNode() ), true );
      
      // if the start pointer is going to execute an exit() instead of an enter() on its trail node, we need to bump up the layer stack an additional step
      var targetLength = this.startPointer.trail.length - ( this.startPointer.isBefore ? 1 : 0 );
      
      while ( pointer.trail.length <= targetLength ) {
        var node = pointer.trail.lastNode();
        if ( node.layerStrategy.hasPreferredLayerType( pointer, this ) ) {
          this.pushPreferredLayerType( node.layerStrategy.getPreferredLayerType( pointer, this ) );
        }
        pointer.trail.addDescendant( this.startPointer.trail.nodes[pointer.trail.length] );
      }
    },
    
    run: function() {
      var builder = this;
      
      // push preferred layers for ancestors of our start pointer
      this.prepareLayerStack();
      
      // console.log( '         stack: ' + _.map( builder.layerTypeStack, function( type ) { return type.name; } ).join( ', ' ) );
      
      builder.startPointer.depthFirstUntil( builder.endPointer, function( pointer ) {
        var node = pointer.trail.lastNode();
        
        if ( pointer.isBefore ) {
          // console.log( 'builder: enter ' + pointer.toString() );
          node.layerStrategy.enter( pointer, builder );
        } else {
          // console.log( 'builder: exit ' + pointer.toString() );
          node.layerStrategy.exit( pointer, builder );
        }
        // console.log( '         stack: ' + _.map( builder.layerTypeStack, function( type ) { return type.name; } ).join( ', ' ) );
      }, false ); // include the endpoints
      
      // special case handling if we are at the 'end' of the scene, so that we create another 'wrapping' boundary
      if ( !this.includesEndTrail ) {
        // console.log( 'builder: not including end trail' );
        this.pendingBoundary.previousEndPointer = builder.endPointer; // TODO: consider implications if we leave this null, to indicate that it is not ended?
        this.layerChange( null );
      }
    },
    
    // allows selfPointer === null at the end if the main iteration's nextSelfTrail === null (i.e. we are at the end of the scene)
    layerChange: function( selfPointer ) {
      this.layerChangePending = false;
      
      var confirmedBoundary = this.pendingBoundary;
      
      confirmedBoundary.nextSelfTrail = selfPointer ? selfPointer.trail.copy() : null;
      
      this.boundaries.push( confirmedBoundary );
      
      this.pendingBoundary = new scenery.LayerBoundary();
      this.pendingBoundary.previousLayerType = confirmedBoundary.nextLayerType;
      this.pendingBoundary.previousSelfTrail = confirmedBoundary.nextSelfTrail;
      // console.log( 'builder:   added boundary' );
    },
    
    /*---------------------------------------------------------------------------*
    * API for layer strategy or other interaction
    *----------------------------------------------------------------------------*/
    
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
