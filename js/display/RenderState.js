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
  require( 'SCENERY/util/Trail' );

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

    },

    // what is our absolute transform relative to (hah)? we assume all transforms up to the last node of this trail have already been applied
    getTransformBaseTrail: function() {

    },

    // whether our backbone child has a CSS transform applied
    isBackboneTransformed: function() {

    }
  } );

  // NOTE: assumes that the trail is not mutable
  RenderState.TestState = function TestState( trail, renderers, isProxy, isUnderCanvasCache, transformBaseTrail ) {
    trail.setImmutable();

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
    this.transformBaseTrail = transformBaseTrail;
    this.nextTransformBaseTrail = transformBaseTrail; // what descendant states will have as their base trail. affected by CSS transformed backbones and single caches
    this.backboneTransformed = false;

    var hints = node.hints || {}; // TODO: reduce allocation here

    if ( !isProxy ) {
      // check if we need a backbone or cache
      if ( node.opacity !== 1 || hints.requireElement || hints.cssTransformBackbone ) {
        this.backbone = true;
        this.backboneTransformed = !!hints.cssTransformBackbone; // for now, only trigger CSS transform if we have the specific hint
        if ( this.backboneTransformed ) {
          // everything under here should not apply transforms from this trail, but only any transforms beneath it
          this.nextTransformBaseTrail = trail;
        }
        this.renderer = scenery.Renderer.DOM; // probably won't be used
        this.nextRenderers = renderers;
      }
      else if ( hints.canvasCache ) {
        if ( combinedBitmask & scenery.bitmaskSupportsCanvas !== 0 ) {
          this.canvasCache = true;
          if ( hints.singleCache ) {
            this.cacheShared = true;
            this.nextTransformBaseTrail = new scenery.Trail();
          }
          this.renderer = scenery.Renderer.Canvas; // TODO: allow SVG (toDataURL) and DOM (direct Canvas)
          this.nextRenderers = [scenery.Renderer.Canvas]; // TODO: full resolution!
        }
        else {
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
        return new RenderState.TestState( trail, this.nextRenderers, true, true, this.nextTransformBaseTrail ); // TODO: allocation
      }
      else {
        return new RenderState.TestState( trail, this.nextRenderers, false, this.isUnderCanvasCache, this.nextTransformBaseTrail ); // TODO: allocation
      }
    },

    getPaintedRenderer: function() {
      return this.renderer;
    },

    getCacheRenderer: function() {
      return this.renderer;
    },

    getTransformBaseTrail: function() {
      return this.transformBaseTrail;
    },

    isBackboneTransformed: function() {
      return this.backboneTransformed;
    }
  };

  return RenderState;
} );
