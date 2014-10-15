// Copyright 2002-2014, University of Colorado Boulder


/**
 * A RenderState represents the state information for a Node needed to determine how descendants are rendered.
 * It extracts all the information necessary from ancestors in a compact form so we can create an effective tree of
 * these.
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
 * NOTE: Trails for transforms are not provided. Instead, inspecting isTransformed and what type of cache should
 * uniquely determine the transformBaseTrail and transformTrail necessary for rendering (and in an efficient way).
 * Not included here for performance (state doesn't need them)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var scenery = require( 'SCENERY/scenery' );
  require( 'SCENERY/display/Renderer' );
  require( 'SCENERY/util/Trail' );

  var emptyObject = {};

  /*
   * @param {Node} node                    The node whose instance will have this state (inspect the hints / properties
   *                                       on this node)
   * @param {Renderer} preferredRenderers  Either 0 (no preference), or a Renderer order bitmask (see Renderer.js)
   * @param {Renderer} svgRenderer            SVG renderer settings to use
   * @param {Renderer} canvasRenderer      Canvas renderer settings to use
   * @param {Renderer} isUnderCanvasCache  Whether we are under any sort of Canvas cache (not including if this node is
   *                                       canvas cached)
   * @param {boolean}  isShared            Whether this is the shared instance tree for a single-cache, instead of a
   *                                       reference to it
   *
   * Potential ways the state can change:
   * - any input changes, specifically including:
   * - node's renderer summary (what types of renderers are allowed below)
   * - node.hints | node.opacity
   */
  var RenderState = scenery.RenderState = function RenderState( node, preferredRenderers, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot ) {
    this.initialize( node, preferredRenderers, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot );
  };
  inherit( Object, RenderState, {
    initialize: function( node, preferredRenderers, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot ) {
      // this should be accurate right now, the pass to update these should have been completed earlier
      var combinedBitmask = node._rendererSummary.bitmask;

      this.preferredRenderers = preferredRenderers;
      this.svgRenderer = svgRenderer;
      this.canvasRenderer = canvasRenderer;
      this.isUnderCanvasCache = isUnderCanvasCache;
      this.isDisplayRoot = isDisplayRoot;

      this.isBackbone = false;
      this.isTransformed = false;
      this.isInstanceCanvasCache = false;
      this.isSharedCanvasCachePlaceholder = false;
      this.isSharedCanvasCacheSelf = false;

      //OHTWO TODO PERFORMANCE: These probably should be 0 (e.g. no renderer), so they are falsy, but that way remain
      // fixednum when set to actual renderers
      this.selfRenderer = null;
      this.groupRenderer = null;
      this.sharedCacheRenderer = null;

      var hints = node._hints || emptyObject;

      //OHTWO TODO: force the same layer/block for these. temporarily we are creating a backbone, but we should be able
      //            to create a composite and have these handled inside layers
      var hasTransparency = node._opacity !== 1 || hints.usesOpacity;
      var hasClip = node._clipArea !== null;

      if ( hints.renderer ) {
        this.preferredRenderers = scenery.Renderer.pushOrderBitmask( this.preferredRenderers, hints.renderer );
      }

      // check if we need a backbone or cache
      // if we are under a canvas cache, we will NEVER have a backbone
      // splits are accomplished just by having a backbone
      if ( isDisplayRoot || ( !isUnderCanvasCache && ( hasTransparency || hasClip || hints.requireElement || hints.cssTransform || hints.split ) ) ) {
        this.isBackbone = true;
        this.isTransformed = isDisplayRoot || !!hints.cssTransform; // for now, only trigger CSS transform if we have the specific hint
        this.groupRenderer = scenery.Renderer.bitmaskDOM | ( hints.forceAcceleration ? scenery.Renderer.bitmaskForceAcceleration : 0 ); // probably won't be used
      }
      else if ( hasTransparency || hasClip || hints.canvasCache ) {
        // everything underneath needs to be renderable with Canvas, otherwise we cannot cache
        assert && assert( ( combinedBitmask & scenery.Renderer.bitmaskCanvas ) !== 0, 'hints.canvasCache provided, but not all node contents can be rendered with Canvas under ' + node.constructor.name );

        if ( hints.singleCache ) {
          // TODO: scale options - fixed size, match highest resolution (adaptive), or mipmapped
          if ( isShared ) {
            this.isUnderCanvasCache = true;
            this.isSharedCanvasCacheSelf = true;

            //OHTWO TODO: Also consider SVG output
            this.sharedCacheRenderer = scenery.Renderer.bitmaskCanvas;
          }
          else {
            // everything underneath needs to guarantee that its bounds are valid
            //OHTWO TODO: We'll probably remove this if we go with the "safe bounds" approach
            assert && assert( ( combinedBitmask & scenery.bitmaskBoundsValid ) !== 0, 'hints.singleCache provided, but not all node contents have valid bounds under ' + node.constructor.name );

            this.isSharedCanvasCachePlaceholder = true;
          }
        }
        else {
          this.isInstanceCanvasCache = true;
          this.isUnderCanvasCache = true;
          this.groupRenderer = scenery.Renderer.bitmaskCanvas; // disallowing SVG here, so we don't have to break up our SVG group structure
        }
      }

      if ( node.isPainted() ) {
        this.setSelfRenderer( node );
      }

      return this;
    },

    setSelfRenderer: function( node ) {
      if ( this.isUnderCanvasCache ) {
        this.selfRenderer = this.canvasRenderer;
      }
      else {
        var success = false;

        // try preferred order if specified
        if ( this.preferredRenderers ) {
          success = this.trySelfRenderer( node, scenery.Renderer.bitmaskOrderFirst( this.preferredRenderers ), 0 ) ||
                    this.trySelfRenderer( node, scenery.Renderer.bitmaskOrderSecond( this.preferredRenderers ), 0 ) ||
                    this.trySelfRenderer( node, scenery.Renderer.bitmaskOrderThird( this.preferredRenderers ), 0 ) ||
                    this.trySelfRenderer( node, scenery.Renderer.bitmaskOrderFourth( this.preferredRenderers ), 0 );
        }

        // fall back to a default order
        success = success ||
                  this.trySelfRenderer( node, scenery.Renderer.bitmaskSVG, this.svgRenderer ) ||
                  this.trySelfRenderer( node, scenery.Renderer.bitmaskCanvas, this.canvasRenderer ) ||
                  this.trySelfRenderer( node, scenery.Renderer.bitmaskDOM, 0 ) ||
                  this.trySelfRenderer( node, scenery.Renderer.bitmaskWebGL, 0 );

        assert && assert( success, 'setSelfRenderer failure?' );
      }
    },

    // returns success of setting this.selfRenderer. renderer is expected to be a 1-bit bitmask, with rendererSpecifics
    // either 0 (autodetect) or a full renderer bitmask
    trySelfRenderer: function( node, renderer, rendererSpecifics ) {
      // if provided renderer === 0, we won't be compatible
      var compatible = !!( node._rendererBitmask & renderer );
      if ( compatible ) {
        if ( !rendererSpecifics ) {
          if ( scenery.Renderer.isCanvas( renderer ) ) {
            rendererSpecifics = this.canvasRenderer;
          }
          else if ( scenery.Renderer.isSVG( renderer ) ) {
            rendererSpecifics = this.svgRenderer;
          }
          else if ( scenery.Renderer.isDOM( renderer ) ) {
            // TODO: decide if CSS transform is to be applied here!
            rendererSpecifics = scenery.Renderer.bitmaskDOM;
          }
          else if ( scenery.Renderer.isWebGL( renderer ) ) {
            // TODO: details?
            rendererSpecifics = scenery.Renderer.bitmaskWebGL;
          }
        }
        assert && assert( rendererSpecifics,
          'Should have renderer specifics by now, otherwise we did not recognize the renderer' );
        this.selfRenderer = rendererSpecifics;
      }
      return compatible;
    },

    getStateForDescendant: function( node ) {
      // TODO: allocation (pool this)
      return RenderState.createFromPool(
        node,

        // default renderer when available
        this.preferredRenderers,

        // default SVG renderer settings
        this.svgRenderer,

        // default Canvas renderer settings
        this.canvasRenderer,

        // isUnderCanvasCache
          this.isUnderCanvasCache || this.isInstanceCanvasCache || this.isSharedCanvasCacheSelf,

        // isShared. No direct descendant is shared, since we create those specially with a new state
        // from createSharedCacheState
        false,

        // isDisplayRoot
        false
      );
    },

    /*
     * Whether we can just update the state on an Instance when changing from this state => otherState.
     * This is generally not possible if there is a change in whether the instance should be a transform root
     * (e.g. backbone/single-cache), so we will have to recreate the instance and its subtree if that is the case.
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
                   ( this.isTransformed ? 'TR ' : '' ) +
                   ( this.selfRenderer ? this.selfRenderer.toString( 16 ) : '-' ) + ',' +
                   ( this.groupRenderer ? this.groupRenderer.toString( 16 ) : '-' ) + ',' +
                   ( this.sharedCacheRenderer ? this.sharedCacheRenderer.toString( 16 ) : '-' ) + ' ';
      return result + ']';
    }
  } );

  RenderState.createRootState = function( node ) {
    var baseState = RenderState.createFromPool(
      node,                           // trail
      0,                              // no preferred renderers
      scenery.Renderer.bitmaskSVG,    // default SVG renderer settings
      scenery.Renderer.bitmaskCanvas, // default Canvas renderer settings
      false,                          // isUnderCanvasCache
      false,                          // isShared
      true                            // isDisplayRoot
    );
    return baseState;
  };

  RenderState.createSharedCacheState = function( node ) {
    var baseState = RenderState.createFromPool(
      node,                             // trail
      0,                                // no preferred renderers (not really necessary, Canvas will be forced anyways)
      null,                             // no SVG renderer settings needed
      scenery.Renderer.bitmaskCanvas,   // default Canvas renderer settings
      true,                             // isUnderCanvasCache
      true,                             // isShared (since we are creating the shared one, not the individual instances referencing it)
      false                             // isDisplayRoot
    );
    return baseState;
  };

  /* jshint -W064 */
  Poolable( RenderState, {
    constructorDuplicateFactory: function( pool ) {
      return function( node, preferredRenderers, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot ) {
        if ( pool.length ) {
          sceneryLog && sceneryLog.RenderState && sceneryLog.RenderState( 'new from pool' );
          return pool.pop().initialize( node, preferredRenderers, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot );
        }
        else {
          sceneryLog && sceneryLog.RenderState && sceneryLog.RenderState( 'new from constructor' );
          return new RenderState( node, preferredRenderers, svgRenderer, canvasRenderer, isUnderCanvasCache, isShared, isDisplayRoot );
        }
      };
    }
  } );

  return RenderState;
} );
