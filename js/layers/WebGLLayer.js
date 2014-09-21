// Copyright 2002-2013, University of Colorado

/**
 * WebGL-backed layer
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var Bounds2 = require( 'DOT/Bounds2' );

  var scenery = require( 'SCENERY/scenery' );

  var Shape = require( 'KITE/Shape' );

  var Layer = require( 'SCENERY/layers/Layer' ); // uses Layer's prototype for inheritance
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/util/Util' );

  // stores CanvasContextWrappers to be re-used
  var canvasContextPool = [];

  // assumes main is wrapped with JQuery
  /*
   *
   */
  scenery.WebGLLayer = function WebGLLayer( args ) {
    sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' constructor' );
    Layer.call( this, args );

    this.dirty = true;

    // TODO: deprecate Scene's backing scale, and handle this on a layer-by-layer option?
    this.backingScale = args.scene.backingScale;
    if ( args.fullResolution !== undefined ) {
      this.backingScale = args.fullResolution ? scenery.Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    }

    this.logicalWidth = this.scene.sceneBounds.width;
    this.logicalHeight = this.scene.sceneBounds.height;

    this.canvas = document.createElement( 'canvas' );

    this.gl = null;
    try {
      this.gl = this.gl = this.canvas.getContext( 'webgl' ) || this.canvas.getContext( 'experimental-webgl' );
      // TODO: check for required extensions
    } catch ( e ) {
      // TODO: handle gracefully
      throw e;
    }
    if ( !this.gl ) {
      throw new Error( 'Unable to load WebGL' );
    }

    this.currentProgram = null; // {MOBIUS/ShaderProgram}

    this.gl.clearColor( 0.0, 0.0, 0.0, 0.0 );

    this.gl.enable( this.gl.DEPTH_TEST );
    this.gl.enable( this.gl.BLEND );
    this.gl.blendFunc( this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA );

    this.canvas.width = this.logicalWidth * this.backingScale;
    this.canvas.height = this.logicalHeight * this.backingScale;
    this.canvas.style.width = this.logicalWidth + 'px';
    this.canvas.style.height = this.logicalHeight + 'px';
    this.canvas.style.position = 'absolute';
    this.canvas.style.left = '0';
    this.canvas.style.top = '0';

    // add this layer on top (importantly, the constructors of the layers are called in order)
    this.$main.append( this.canvas );

    this.scene = args.scene;

    this.isWebGLLayer = true;
  };
  var WebGLLayer = scenery.WebGLLayer;

  inherit( Layer, WebGLLayer, {

    render: function( scene, args ) {

    },

    dispose: function() {
      Layer.prototype.dispose.call( this );

      this.canvas.parentNode.removeChild( this.canvas );
    },

    applyTransformationMatrix: function( matrix ) {

    },

    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      Layer.prototype.reindex.call( this, zIndex );

      if ( this.zIndex !== zIndex ) {
        this.canvas.style.zIndex = zIndex;
        this.zIndex = zIndex;
      }
      return zIndex + 1;
    },

    pushClipShape: function( shape ) {

    },

    popClipShape: function() {

    },

    getSVGString: function() {
      // TODO: probably broken
      return '<image xmlns:xlink="' + scenery.xlinkns + '" xlink:href="' + this.canvas.toDataURL() + '" x="0" y="0" height="' + this.canvas.height + 'px" width="' + this.canvas.width + 'px"/>';
    },

    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      context.drawImage( this.canvas, 0, 0 );
    },

    addInstance: function( instance ) {
      var trail = instance.trail;

      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' addInstance: ' + trail.toString() );
      Layer.prototype.addInstance.call( this, instance );

      // TODO
    },

    removeInstance: function( instance ) {
      var trail = instance.trail;

      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' removeInstance: ' + trail.toString() );
      Layer.prototype.removeInstance.call( this, instance );

      // TODO
    },

    getName: function() {
      return 'webgl';
    },

    markWebGLDirty: function() {
      this.dirty = true;
    },

    /*---------------------------------------------------------------------------*
    * Events from Instances
    *----------------------------------------------------------------------------*/

    notifyVisibilityChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyVisibilityChange: ' + instance.trail.toString() );
      // old paint taken care of in notifyBeforeSubtreeChange()

      this.markWebGLDirty();
    },

    notifyOpacityChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyOpacityChange: ' + instance.trail.toString() );
      // old paint taken care of in notifyBeforeSubtreeChange()

      this.markWebGLDirty();
    },

    notifyClipChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyClipChange: ' + instance.trail.toString() );
      // old paint taken care of in notifyBeforeSubtreeChange()

      this.markWebGLDirty();
    },

    // only a painted trail under this layer
    notifyBeforeSelfChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyBeforeSelfChange: ' + instance.trail.toString() );

      this.markWebGLDirty();
    },

    notifyBeforeSubtreeChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyBeforeSubtreeChange: ' + instance.trail.toString() );

      this.markWebGLDirty();
    },

    // only a painted trail under this layer
    notifyDirtySelfPaint: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyDirtySelfPaint: ' + instance.trail.toString() );

      this.markWebGLDirty();
    },

    notifyDirtySubtreePaint: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyDirtySubtreePaint: ' + instance.trail.toString() );

      this.markWebGLDirty();
    },

    notifyTransformChange: function( instance ) {
      // sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyTransformChange: ' + instance.trail.toString() );
      // TODO: how best to mark this so if there are multiple 'movements' we don't get called more than needed?
      // this.canvasMarkSubtree( instance );

      this.markWebGLDirty();
    },

    // only a painted trail under this layer (for now)
    notifyBoundsAccuracyChange: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'WebGLLayer #' + this.id + ' notifyBoundsAccuracyChange: ' + instance.trail.toString() );

      this.markWebGLDirty();
    }
  } );

  return WebGLLayer;
} );


