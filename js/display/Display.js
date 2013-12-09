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
  require( 'SCENERY/display/BackboneBlock' );
  require( 'SCENERY/display/DisplayInstance' );
  require( 'SCENERY/display/RenderState' );
  require( 'SCENERY/layers/Renderer' );
  
  scenery.Display = function Display( rootNode ) {
    this._rootNode = rootNode;
    this._domElement = null; // TODO: potentially allow immediate export of this?
    this._sharedCanvasInstances = {}; // map from Node ID to DisplayInstance, for fast lookup
    this._baseInstance = null; // will be filled with the root DisplayInstance
    
    this._frameId = 0; // incremented for every rendered frame
  };
  var Display = scenery.Display;
  
  // recursively compute the bitmask intersection (bitwise AND) for a node and all of its children, and store it to that node's _subtreeRendererBitmask
  // TODO: partial updates (and speed for that)
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
  function connectDrawables( a, b ) {
    a.nextDrawable = b;
    b.previousDrawable = a;
  }
  
  function createInstance( display, trail, state, parentInstance ) {
    var instance = new scenery.DisplayInstance( this, trail );
    
    var isSharedCache = state.isCanvasCache && state.isCacheShared;
    
    var node = trail.lastNode();
    instance.state = state;
    instance.parent = parentInstance;
    
    if ( isSharedCache ) {
      var instanceKey = trail.lastNode().getId();
      var sharedInstance = display._sharedCanvasInstances[instanceKey];
      
      // TODO: assert state is the same?
      // TODO: increment reference counting?
      if ( !sharedInstance ) {
        var sharedNode = trail.lastNode();
        sharedInstance = createInstance( display, new scenery.Trail( sharedNode ), scenery.RenderState.RegularState.createSharedCacheState( sharedNode ), null );
        display._sharedCanvasInstances[instanceKey] = sharedInstance;
        // TODO: reference counting?
        
        // TODO: sharedInstance.isTransformed?
      }
      
      // TODO: do something with the sharedInstance!
      var sharedCacheRenderer = state.sharedCacheRenderer;
      instance.sharedCacheDrawable = // TODO create
    } else {
      var currentDrawable = null;
      
      if ( node.isPainted() ) {
        // dynamic import
        var Renderer = scenery.Renderer;
        
        var selfRenderer = state.selfRenderer;
        var selfRendererType = selfRenderer & Renderer.bitmaskRendererArea;
        if ( selfRendererType === Renderer.bitmaskCanvas ) {
          instance.selfDrawable = new scenery.CanvasSelfDrawable( trail, selfRenderer, instance );
        } else if ( selfRendererType === Renderer.bitmaskSVG ) {
          instance.selfDrawable = new scenery.SVGSelf( trail, selfRenderer, instance );
        } else if ( selfRendererType === Renderer.bitmaskDOM ) {
          instance.selfDrawable = new scenery.DOMSelfDrawable( trail, selfRenderer, instance );
        } else {
          // assert so that it doesn't compile down to a throw (we want this function to be optimized)
          assert && assert( 'Unrecognized renderer, maybe we don\'t support WebGL yet?: ' + selfRenderer );
        }
        currentDrawable = instance.selfDrawable;
      }
      
      var children = trail.lastNode().children;
      var numChildren = children.length;
      for ( var i = 0; i < numChildren; i++ ) {
        // create a child instance
        var child = children[i];
        var childInstance = createInstance( display, trail.copy().addDescendant( child, i ), state.getStateForDescendant( child ), instance );
        instance.appendInstance( childInstance );
        
        // figure out what the first and last drawable should be hooked into for the child
        var firstChildDrawable = null;
        var lastChildDrawable = null;
        if ( childInstance.groupDrawable ) {
          // if there is a group (e.g. non-shared cache or backbone), use it
          firstChildDrawable = lastChildDrawable = childInstance.groupDrawable;
        } else if ( childInstance.sharedCacheDrawable ) {
          // if there is a shared cache drawable, use it
          firstChildDrawable = lastChildDrawable = childInstance.sharedCacheDrawable;
        } else if ( childInstance.firstDrawable ) {
          // otherwise, if they exist, pick the node's first/last directly
          assert && assert( childInstance.lastDrawable, 'Any display instance with firstDrawable should also have lastDrawable' );
          firstChildDrawable = childInstance.firstDrawable;
          lastChildDrawable = childInstance.lastDrawable;
        }
        
        // if there are any drawables for that child, link them up in our linked list
        if ( firstChildDrawable ) {
          if ( currentDrawable ) {
            // there is already an end of the linked list, so just append to it
            connectDrawables( currentDrawable, firstChildDrawable );
          } else {
            // start out the linked list
            instance.firstDrawable = firstChildDrawable;
          }
          // update the last drawable of the linked list
          currentDrawable = lastChildDrawable;
        }
      }
      if ( currentDrawable !== null ) {
        // finish setting up references to the linked list (now firstDrawable and lastDrawable should be set properly)
        instance.lastDrawable = currentDrawable;
      }
      
      var groupRenderer = state.groupRenderer;
      if ( state.isBackbone ) {
        assert && assert( !isCanvasCache, 'For now, disallow an instance being a backbone and a canvas cache, since it has no performance benefits' );
        
        var backbone = new scenery.BackboneBlock( instance );
        instance.block = backbone;
        instance.groupDrawable = backbone.getDOMDrawable();
        instance.isTransformed = state.isBackboneTransformed; // we allow non-transformed backbones, for things like filters (where we don't want a CSS transform for non-integral coordinates)
      } else if ( state.isCanvasCache ) {
        instance.groupDrawable = // TODO create non-shared cache, use groupRenderer
      }
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
      this._frameId++;
      
      // validate bounds for everywhere that could trigger bounds listeners. we want to flush out any changes, so that we can call validateBounds()
      // from code below without triggering side effects (we assume that we are not reentrant).
      this._rootNode.validateWatchedBounds();
      
      // compute updated _subtreeRendererBitmask for every Node // TODO: add and use dirty flag for this, and decide how the flags get set!
      recursiveUpdateRendererBitmask( this._rootNode );
      
      throw new Error( 'TODO: replace with actual stitching' );
      this._baseInstance = createInstance( this, baseTrail, scenery.RenderState.RegularState.createRootState( this._rootNode ), null );
      
      throw new Error( 'TODO: pre-repaint phase (first transform-notify by traversing all transform roots)' );
      
      throw new Error( 'TODO: repaint phase (painting)' );
      
      throw new Error( 'TODO: update cursor' );
    }
  } );
  
  return Display;
} );
