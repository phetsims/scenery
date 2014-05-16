// Copyright 2002-2014, University of Colorado

/**
 * A RenderState represents the state information for a Node needed to determine how descendants are rendered.
 * It extracts all the information necessary from ancestors in a compact form so we can create an effective tree of these.
 *
 * API for RenderState:
 * {
 *   isBackbone: Boolean
 *   isTransformed: Boolean
 *   isInstanceCanvasCache: Boolean
 *   isSharedCanvasCachePlaceholder: Boolean
 *   isSharedCanvasCacheSelf: Boolean
 *   selfRenderer: Renderer
 *   groupRenderer: Renderer
 *   sharedCacheRenderer: Renderer
 *   getStateForDescendant: function( trail ) : RenderState
 * }
 *
 * NOTE: Trails for transforms are not provided. Instead, inspecting isTransformed and what type of cache should uniquely determine
 *       the transformBaseTrail and transformTrail necessary for rendering (and in an efficient way). Not included here for performance (state doesn't need them)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';
  
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/layers/Renderer' );
  require( 'SCENERY/util/Trail' );
  
  var RenderState = scenery.RenderState = {};
  
  var emptyObject = {};
  
  /*
   * {param} node               Node      The node whose instance will have this state (inspect the hints / properties on this node)
   * {param} svgRenderer        Renderer  SVG renderer settings to use
   * {param} canvasRenderer     Renderer  Canvas renderer settings to use
   * {param} isUnderCanvasCache Boolean   Whether we are under any sort of Canvas cache (not including if this node is canvas cached)
   * {param} isShared           Boolean   Whether this is the shared instance tree for a single-cache, instead of a reference to it
   *
   * Potential ways the state can change:
   * - any input changes, specifically including:
   * - node's renderer summary (what types of renderers are allowed below)
   * - node.hints | node.opacity
   */
  RenderState.RegularState = function RegularState( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot ) {
    this.initialize( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot );
  };
  RenderState.RegularState.prototype = {
    constructor: RenderState.RegularState,
    
    initialize: function( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot ) {
      // this should be accurate right now, the pass to update these should have been completed earlier
      var combinedBitmask = node._rendererSummary.bitmask;
      
      this.svgRenderer = svgRenderer;
      this.canvasRenderer = canvasRenderer;
      this.isUnderCanvasCache = isUnderCanvasCache;
      this.isDisplayRoot = isDisplayRoot;
      
      this.isBackbone = false;
      this.isTransformed = false;
      this.isInstanceCanvasCache = false;
      this.isSharedCanvasCachePlaceholder = false;
      this.isSharedCanvasCacheSelf = false;
      
      //OHTWO TODO PERFORMANCE: These probably should be 0 (e.g. no renderer), so they are falsy, but that way remain fixednum when set to actual renderers
      this.selfRenderer = null;
      this.groupRenderer = null;
      this.sharedCacheRenderer = null;
      
      var hints = node._hints || emptyObject;
      
      var isTransparent = node.opacity !== 1;
      
      // check if we need a backbone or cache
      // if we are under a canvas cache, we will NEVER have a backbone
      // splits are accomplished just by having a backbone
      if ( isDisplayRoot || ( !isUnderCanvasCache && ( isTransparent || hints.requireElement || hints.cssTransform || hints.split ) ) ) {
        this.isBackbone = true;
        this.isTransformed = isDisplayRoot || !!hints.cssTransform; // for now, only trigger CSS transform if we have the specific hint
        this.groupRenderer = scenery.Renderer.bitmaskDOM | ( hints.forceAcceleration ? scenery.Renderer.bitmaskForceAcceleration : 0 ); // probably won't be used
      } else if ( isTransparent || hints.canvasCache ) {
        // everything underneath needs to be renderable with Canvas, otherwise we cannot cache
        assert && assert( ( combinedBitmask & scenery.Renderer.bitmaskCanvas ) !== 0, 'hints.canvasCache provided, but not all node contents can be rendered with Canvas under ' + node.constructor.name );
        
        if ( hints.singleCache ) {
          // TODO: scale options - fixed size, match highest resolution (adaptive), or mipmapped
          if ( isShared ) {
            this.isUnderCanvasCache = true;
            this.isSharedCanvasCacheSelf = true;
            
            //OHTWO TODO: Also consider SVG output
            this.sharedCacheRenderer = scenery.Renderer.bitmaskCanvas;
          } else {
            // everything underneath needs to guarantee that its bounds are valid
            //OHTWO TODO: We'll probably remove this if we go with the "safe bounds" approach
            assert && assert( ( combinedBitmask & scenery.bitmaskBoundsValid ) !== 0, 'hints.singleCache provided, but not all node contents have valid bounds under ' + node.constructor.name );
            
            this.isSharedCanvasCachePlaceholder = true;
          }
        } else {
          this.isInstanceCanvasCache = true;
          this.isUnderCanvasCache = true;
          this.groupRenderer = scenery.Renderer.bitmaskCanvas; // disallowing SVG here, so we don't have to break up our SVG group structure
        }
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
        } else if ( ( scenery.Renderer.bitmaskDOM & node._rendererBitmask ) !== 0 ) {
          // TODO: decide if CSS transform is to be applied here!
          this.selfRenderer = scenery.Renderer.bitmaskDOM;
        } else {
          throw new Error( 'unsupported renderer, something wrong in RenderState' );
        }
      }
      
      return this;
    },
    
    getStateForDescendant: function( node ) {
      // TODO: allocation (pool this)
      return RenderState.RegularState.createFromPool(
        node,
        
        // default SVG renderer settings
        this.svgRenderer,
        
        // default Canvas renderer settings
        this.canvasRenderer,
        
        // isUnderCanvasCache
        this.isUnderCanvasCache || this.isInstanceCanvasCache || this.isSharedCanvasCacheSelf,
        
        // isShared. No direct descendant is shared, since we create those specially with a new state from createSharedCacheState
        false,
        
        // isDisplayRoot
        false
      );
    },
    
    /*
     * Whether we can just update the state on a DisplayInstance when changing from this state => otherState.
     * This is generally not possible if there is a change in whether the instance should be a transform root (e.g. backbone/single-cache),
     * so we will have to recreate the instance and its subtree if that is the case.
     */
    isInstanceCompatibleWith: function( otherState ) {
      return this.isTransformed === otherState.isTransformed && //OHTWO TODO: allow mutating based on this change
             this.isSharedCanvasCachePlaceholder === otherState.isSharedCanvasCachePlaceholder; //OHTWO TODO: allow mutating based on this change
    },
    
    toString: function() {
      var result = 'S[ ' +
                   ( this.isDisplayRoot ? 'displayRoot ' : '' ) +
                   ( this.isBackbone ? 'backbone ' : '' ) +
                   ( this.isInstanceCanvasCache ? 'instanceCache ' : '' ) +
                   ( this.isSharedCanvasCachePlaceholder ? 'sharedCachePlaceholder ' : '' ) +
                   ( this.isSharedCanvasCacheSelf ? 'sharedCacheSelf ' : '' ) +
                   ( this.isTransformed ? 'TR' : '' ) +
                   ( this.selfRenderer ? this.selfRenderer.toString( 16 ) : '-' ) + ',' +
                   ( this.groupRenderer ? this.groupRenderer.toString( 16 ) : '-' ) + ',' +
                   ( this.sharedCacheRenderer ? this.sharedCacheRenderer.toString( 16 ) : '-' ) + ' ';
      return result + ']';
    }
  };
  
  RenderState.RegularState.createRootState = function( node ) {
    var baseState = RenderState.RegularState.createFromPool(
      node,                           // trail
      scenery.Renderer.bitmaskSVG,    // default SVG renderer settings
      scenery.Renderer.bitmaskCanvas, // default Canvas renderer settings
      false,                          // isUnderCanvasCache
      false,                          // isShared
      true                            // isDisplayRoot
    );
    return baseState;
  };
  
  RenderState.RegularState.createSharedCacheState = function( node ) {
    var baseState = RenderState.RegularState.createFromPool(
      node,                             // trail
      null,                             // no SVG renderer settings needed
      scenery.Renderer.bitmaskCanvas,   // default Canvas renderer settings
      true,                             // isUnderCanvasCache
      true,                             // isShared (since we are creating the shared one, not the individual instances referencing it)
      false                             // isDisplayRoot
    );
    return baseState;
  };
  
  /* jshint -W064 */
  Poolable( RenderState.RegularState, {
    constructorDuplicateFactory: function( pool ) {
      return function( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot ) {
        if ( pool.length ) {
          return pool.pop().initialize( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot );
        } else {
          return new RenderState.RegularState( node, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot );
        }
      };
    }
  } );
  
  return RenderState;
} );
