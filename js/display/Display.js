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
  
  scenery.Display = function Display( rootNode ) {
    this._rootNode = rootNode;
    this._domElement = null; // TODO: potentially allow immediate export of this?
    this._sharedCanvasInstances = {}; // map from Node ID to DisplayInstance, for fast lookup
    this._instanceTree = null; // will be filled with the root DisplayInstance
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
  
  function freshInstance( display, trail, ancestorState ) {
    var state = ancestorState.getStateForDescendant( trail );
    if ( state.isCanvasShared() ) {
      var instanceKey = trail.lastNode().getId();
      var sharedInstance = display._sharedCanvasInstances[instanceKey];
      if ( sharedInstance ) {
        // TODO: assert state is the same?
        // TODO: increment reference counting?
        return sharedInstance;
      } else {
        var instance = setupInstance( display, new scenery.Trail( trail.lastNode() ), state );
        // TODO: increment reference counting?
        display._sharedCanvasInstances[instanceKey] = instance;
        return instance;
      }
    } else {
      // not shared
      return setupInstance( display, trail, state );
    }
  }
  
  function setupInstance( display, trail, state ) {
    var instance = new scenery.DisplayInstance( trail );
    instance.state = state;
    var children = trail.lastNode().children;
    var numChildren = children.length;
    for ( var i = 0; i < numChildren; i++ ) {
      instance.appendInstance( freshInstance( display, trail.copy().addDescendant( children[i], i ) ) );
    }
    return instance;
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
    }
  } );
  
  return Display;
} );
