// Copyright 2002-2013, University of Colorado

/**
 * A persistent display of a specific Node and its descendants
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/display/DisplayInstance' );
  require( 'SCENERY/display/RenderState' );
  require( 'SCENERY/layers/Renderer' );
  
  scenery.Display = function Display( rootNode ) {
    this._rootNode = rootNode;
    this._domElement = null; // TODO: potentially allow immediate export of this?
    this._sharedCanvasInstances = {}; // map from Node ID to DisplayInstance, for fast lookup
    this._baseInstance = null; // will be filled with the root DisplayInstance
  };
  var Display = scenery.Display;
  
  // recursively compute the bitmask intersection (bitwise AND) for a node and all of its children, and store it to that node's _subtreeRendererBitmask
  function recursiveUpdateRendererBitmask( node ) {
    var bitmask = scenery.bitmaskAll;
    bitmask &= node._rendererBitmask;
    
    // include all children
    var children = node._children;
    var numChildren = children.length;
    for ( var i = 0; i < numChildren; i++ ) {
      bitmask &= recursiveUpdateRendererBitmask( children[i] );
    }
    
    node._subtreeRendererBitmask = bitmask;
    return bitmask; // return the bitmask so we have direct access at the call site
  }
  
  // display instance linked list ops
  function connectInstances( a, b ) {
    a.nextPainted = b;
    b.previousPainted = a;
  }
  
  function createBackbone( display, trail, state ) {
    var blockInstance = new scenery.DisplayInstance( trail );
    blockInstance.state = state;
    blockInstance.renderer = scenery.Renderer.DOM;
    
    createProxyInstance( display, trail, state, blockInstance );
    return blockInstance;
  }
  
  function createCanvasCache( display, trail, state ) {
    var blockInstance = new scenery.DisplayInstance( trail );
    blockInstance.state = state;
    blockInstance.renderer = state.getCacheRenderer();
    
    createProxyInstance( display, trail, state, blockInstance );
    return blockInstance;
  }
  
  function createSharedCanvasCache( display, trail, state ) {
    var instanceKey = trail.lastNode().getId();
    var sharedInstance = display._sharedCanvasInstances[instanceKey];
    if ( sharedInstance ) {
      // TODO: assert state is the same?
      // TODO: increment reference counting?
      return sharedInstance;
    } else {
      var blockInstance = new scenery.DisplayInstance( new scenery.Trail( trail.lastNode() ) );
      blockInstance.state = state;
      blockInstance.renderer = state.getCacheRenderer();
      
      createProxyInstance( display, trail, state, blockInstance );
      // TODO: increment reference counting?
      display._sharedCanvasInstances[instanceKey] = blockInstance;
      return blockInstance;
    }
  }
  
  // For when we have a block/stub instance (for a backbone/cache), and we want an instance for the same trail, but to render itself and its subtree.
  // Basically, this involves another getStateForDescendant (called in createInstance), and for now we set up proxy variables
  function createProxyInstance( display, trail, state, blockInstance ) {
    var instance = createInstance( display, trail, state );
    // TODO: better way of handling this?
    blockInstance.proxyChild = instance;
    instance.proxyParent = blockInstance;
    
    blockInstance.firstPainted = blockInstance;
    blockInstance.lastPainted = blockInstance;
    return instance; // if we need it
  }
  
  function createSplitInstance() {
    return new scenery.DisplayInstance( null ); // null trail
  }
  
  function createInstance( display, trail, ancestorState ) {
    var state = ancestorState.getStateForDescendant( trail );
    if ( state.isBackbone() ) {
      return createBackbone( display, trail, state );
    } else if ( state.isCanvasCache() ) {
      return state.isCacheShared() ? createSharedCanvasCache( display, trail, state ) : createCanvasCache( display, trail, state );
    } else {
      var instance = new scenery.DisplayInstance( trail );
      var node = trail.lastNode();
      instance.state = state;
      instance.renderer = node.isPainted() ? state.getPaintedRenderer() : null;
      
      var currentPaintedInstance = null;
      if ( instance.isEffectivelyPainted() ) {
        currentPaintedInstance = instance;
        instance.firstPainted = instance;
      }
      
      var children = trail.lastNode().children;
      var numChildren = children.length;
      for ( var i = 0; i < numChildren; i++ ) {
        var childInstance = createInstance( display, trail.copy().addDescendant( children[i], i ), state );
        instance.appendInstance( childInstance );
        if ( childInstance.firstPainted ) {
          assert && assert( childInstance.lastPainted, 'Any display instance with firstPainted should also have lastPainted' );
          
          if ( currentPaintedInstance ) {
            connectInstances( currentPaintedInstance, childInstance.firstPainted );
          } else {
            instance.firstPainted = childInstance.firstPainted;
          }
          currentPaintedInstance = childInstance.lastPainted;
        }
      }
      
      if ( currentPaintedInstance !== null ) {
        instance.lastPainted = currentPaintedInstance;
      }
      
      if ( state.requestsSplit() ) {
        if ( instance.firstPainted ) {
          var beforeSplit = createSplitInstance();
          var afterSplit = createSplitInstance();
          connectInstances( beforeSplit, instance.firstPainted );
          connectInstances( instance.lastPainted, afterSplit );
          instance.firstPainted = beforeSplit;
          instance.lastPainted = afterSplit;
        } else {
          instance.firstPainted = instance.lastPainted = createSplitInstance();
        }
      }
      
      return instance;
    }
  }
  
  inherit( Object, Display, {
    getRootNode: function() {
      return this._rootNode;
    },
    get rootNode() { return this.getRootNode(); },
    
    // NOTE: to be replaced with a full stitching/update version
    buildTemporaryDisplay: function() {
      // compute updated _subtreeRendererBitmask for every Node // TODO: add and use dirty flag for this, and decide how the flags get set!
      recursiveUpdateRendererBitmask( this._rootNode );
      
      var baseTrail = new scenery.Trail( this._rootNode );
      var baseState = new scenery.RenderState.TestState( baseTrail, [
        scenery.Renderer.DOM,
        scenery.Renderer.Canvas,
        scenery.Renderer.SVG,
        new scenery.Trail()
      ], false, false );
      this._baseInstance = createBackbone( this, baseTrail, baseState );
    }
  } );
  
  return Display;
} );
