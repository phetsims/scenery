// Copyright 2002-2013, University of Colorado

/**
 * API for RenderState
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/layers/Renderer' );
  
  scenery.RenderState = function RenderState() {
    
  };
  var RenderState = scenery.RenderState;
  
  inherit( Object, RenderState, {
    isBackbone: function() {
      return false;
    },
    
    isCanvasCache: function() {
      return false;
    },
    
    isCacheShared: function() {
      return false;
    },
    
    requestsSplit: function() {
      return false;
    },
    
    getStateForDescendant: function( trail ) {
      // new state
    },
    
    getPaintedRenderer: function() {
      
    },
    
    // renderer for the (Canvas) cache
    getCacheRenderer: function() {
      
    }
  } );
  
  RenderState.TestState = function TestState( trail, renderers, isProxy, isUnderCanvasCache ) {
    var node = trail.lastNode();
    
    // this should be accurate right now, the pass to update these should have been completed earlier
    var combinedBitmask = node._subtreeRendererBitmask;
    
    this.trail = trail;
    this.renderers = renderers;
    this.isProxy = isProxy;
    this.isUnderCanvasCache;
    
    this.nextRenderers = null; // will be filled with Array?
    
    this.backbone = false;
    this.canvasCache = false;
    this.cacheShared = false;
    this.splits = false;
    this.renderer = null;
    this.cacheRenderer = null;
    
    var hints = node.hints || {}; // TODO: reduce allocation here
    
    if ( !isProxy ) {
      // check if we need a backbone or cache
      if ( node.opacity !== 1 || hints.requireElement ) {
        this.backbone = true;
        this.renderer = scenery.Renderer.DOM; // probably won't be used
        this.nextRenderers = renderers;
      } else if ( hints.canvasCache ) {
        if ( combinedBitmask & scenery.bitmaskSupportsCanvas !== 0 ) {
          this.canvasCache = true;
          if ( hints.singleCache ) {
            this.cacheShared = true;
          }
          this.renderer = scenery.Renderer.Canvas; // TODO: allow SVG (toDataURL) and DOM (direct Canvas)
          this.nextRenderers = [scenery.Renderer.Canvas]; // TODO: full resolution!
        } else {
          assert && assert( false, 'Attempting to canvas cache when nodes underneath can\'t be rendered with Canvas' );
        }
      }
    }
    
    if ( !this.backbone && !this.canvasCache ) {
      if ( hints.layerSplit ) {
        this.splits = true;
      }
      
      // if a node isn't painted (and no backbone/cache), we'll leave the renderer as null
      if ( node.isPainted() ) {
        // pick the top-most renderer that will work
        for ( var i = renderers.length - 1; i >= 0; i-- ) {
          var renderer = renderers[i];
          if ( renderer.bitmask & node._rendererBitmask !== 0 ) {
            this.renderer = renderer;
            break;
          }
        }
      }
      
      this.nextRenderers = renderers;
    }
  };
  RenderState.TestState.prototype = {
    constructor: RenderState.TestState,
    
    isBackbone: function() {
      return this.backbone;
    },
    
    isCanvasCache: function() {
      return this.canvasCache;
    },
    
    isCacheShared: function() {
      return this.cacheShared;
    },
    
    requestsSplit: function() {
      return this.splits;
    },
    
    getStateForDescendant: function( trail ) {
      if ( this.backbone || this.canvasCache ) {
        // proxy instance
        assert && assert( trail === this.trail, 'backbone/cache trail should be passed in again for the proxy instance' );
        // TODO: full resolution handling
        return new RenderState.TestState( trail, this.nextRenderers, true, true ); // TODO: allocation
      } else {
        return new RenderState.TestState( trail, this.nextRenderers, false, this.isUnderCanvasCache ); // TODO: allocation
      }
    },
    
    getPaintedRenderer: function() {
      return this.renderer;
    },
    
    getCacheRenderer: function() {
      return this.renderer;
    }
  };
  
  return RenderState;
} );
