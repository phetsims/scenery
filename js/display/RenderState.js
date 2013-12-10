// Copyright 2002-2013, University of Colorado

/**
 * A RenderState represents the state information for a Node needed to determine how descendants are rendered.
 * It extracts all the information necessary from ancestors in a compact form so we can create an effective tree of these.
 *
 * API for RenderState:
 * {
 *   isBackbone: Boolean
 *   isCanvasCache: Boolean
 *   isCacheShared: Boolean
 *   isTransformed: Boolean
 *   selfRenderer: Renderer
 *   groupRenderer: Renderer
 *   sharedCacheRenderer: Renderer
 *   getStateForDescendant: function( trail ) : RenderState
 * }
 *
 * NOTE: Trails for transforms are not provided. Instead, inspecting isTransformed and what type of cache should uniquely determine
 *       the transformBaseTrail and transformTrail necessary for rendering (and in an efficient way). Not included here for performance (state doesn't need them)
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/layers/Renderer' );
  require( 'SCENERY/util/Trail' );
  
  var RenderState = scenery.RenderState = {};
  
  /*
   * {param} node               Node      The node whose instance will have this state (inspect the hints / properties on this node)
   * {param} svgRenderer        Renderer  SVG renderer settings to use
   * {param} canvasRenderer     Renderer  Canvas renderer settings to use
   * {param} isUnderCanvasCache Boolean   Whether we are under any sort of Canvas cache (not including if this node is canvas cached)
   * {param} isShared           Boolean   Whether this is the shared instance tree for a single-cache, instead of a reference to it
   */
  RenderState.RegularState = function RegularState( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared ) {
    // this should be accurate right now, the pass to update these should have been completed earlier
    var combinedBitmask = node._rendererSummary.bitmask;
    
    this.svgRenderer = svgRenderer;
    this.canvasRenderer = canvasRenderer;
    this.isUnderCanvasCache;
    
    this.isBackbone = false;
    this.isTransformed = false;
    this.isCanvasCache = false;
    this.isCacheShared = false;
    
    this.selfRenderer = null;
    this.groupRenderer = null;
    this.sharedCacheRenderer = null;
    
    var hints = node.hints || {}; // TODO: reduce allocation here
    
    var isTransparent = node.opacity !== 1;
    
    // check if we need a backbone or cache
    // if we are under a canvas cache, we will NEVER have a backbone
    // splits are accomplished just by having a backbone
    if ( !isUnderCanvasCache && ( isTransparent || hints.requireElement || hints.cssTransformBackbone || hints.split ) ) {
      this.isBackbone = true;
      this.isTransformed = !!hints.cssTransformBackbone; // for now, only trigger CSS transform if we have the specific hint
      this.groupRenderer = scenery.Renderer.bitmaskDOM | ( hints.forceAcceleration ? scenery.Renderer.bitmaskForceAcceleration : 0 ); // probably won't be used
    } else if ( isTransparent || hints.canvasCache ) {
      // everything underneath needs to be renderable with Canvas, otherwise we cannot cache
      assert && assert( ( combinedBitmask & scenery.bitmaskSupportsCanvas ) !== 0, 'hints.canvasCache provided, but not all node contents can be rendered with Canvas under ' + node.constructor.name );
      
      // TODO: handling of transformed caches differently than aligned caches?
      
      this.isCanvasCache = true;
      if ( hints.singleCache && !isShared ) {
        // TODO: scale options - fixed size, match highest resolution (adaptive), or mipmapped
        
        // everything underneath needs to guarantee that its bounds are valid
        assert && assert( ( combinedBitmask & scenery.bitmaskBoundsValid ) !== 0, 'hints.singleCache provided, but not all node contents have valid bounds under ' + node.constructor.name );
        this.isCacheShared = true;
        this.isTransformed = true;
      }
      this.selfRenderer = scenery.Renderer.bitmaskCanvas; // TODO: allow SVG (toDataURL) and DOM (direct Canvas)
    }
    
    if ( node.isPainted() ) {
      // TODO: figure out preferred rendering order
      // TODO: many more things to consider here for performance
      // TODO: performance (here)
      if ( isUnderCanvasCache ) {
        this.selfRenderer = canvasRenderer;
      } else if ( svgRenderer && ( svgRenderer & node._rendererBitmask ) !== 0 ) {
        this.selfRenderer = svgRenderer;
      } else if ( canvasRenderer && ( canvasRenderer & node._rendererBitmask ) !== 0 ) {
        this.selfRenderer = canvasRenderer;
      } else if ( scenery.bitmaskSupportsDOM & node._rendererBitmask !== 0 ) {
        // TODO: decide if CSS transform is to be applied here!
        this.selfRenderer = scenery.bitmaskSupportsDOM;
      } else {
        throw new Error( 'unsupported renderer, something wrong in RenderState' );
      }
      
      // TODO: remove this later
      this.selfRenderer = scenery.bitmaskSupportsDOM;
    }
  };
  RenderState.RegularState.prototype = {
    constructor: RenderState.RegularState,
    
    getStateForDescendant: function( node ) {
      // TODO: allocation (pool this)
      return new RenderState.RegularState(
        node,
        
        // default SVG renderer settings
        this.svgRenderer,
        
        // default Canvas renderer settings
        this.canvasRenderer,
        
        // isUnderCanvasCache
        this.isUnderCanvasCache || this.isCanvasCache,
        
        // isShared. No direct descendant is shared, since we create those specially with a new state from createSharedCacheState
        false
      );
    }
  };
  
  RenderState.RegularState.createRootState = function( node ) {
    var baseState = new RenderState.RegularState(
      node,                   // trail
      scenery.Renderer.bitmaskSVG,    // default SVG renderer settings
      scenery.Renderer.bitmaskCanvas, // default Canvas renderer settings
      false,                  // isUnderCanvasCache
      false                   // isShared
    );
    return baseState;
  };
  
  RenderState.RegularState.createSharedCacheState = function( node ) {
    var baseState = new RenderState.RegularState(
      node,                     // trail
      null,                     // no SVG renderer settings needed
      scenery.Renderer.bitmaskCanvas,   // default Canvas renderer settings
      true,                     // isUnderCanvasCache
      true                      // isShared (since we are creating the shared one, not the individual instances referencing it)
    );
    return baseState;
  };
  
  return RenderState;
} );
