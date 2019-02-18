// Copyright 2013-2018, University of Colorado Boulder

/**
 * Handles a visual Canvas layer of drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var CanvasContextWrapper = require( 'SCENERY/util/CanvasContextWrapper' );
  var cleanArray = require( 'PHET_CORE/cleanArray' );
  var FittedBlock = require( 'SCENERY/display/FittedBlock' );
  var inherit = require( 'PHET_CORE/inherit' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Poolable = require( 'PHET_CORE/Poolable' );
  var Renderer = require( 'SCENERY/display/Renderer' );
  var scenery = require( 'SCENERY/scenery' );
  var Util = require( 'SCENERY/util/Util' );
  var Vector2 = require( 'DOT/Vector2' );

  var scratchMatrix = new Matrix3();
  var scratchMatrix2 = new Matrix3();

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param {Display} display
   * @param {number} renderer - See Renderer.js for more information
   * @param {Instance} transformRootInstance
   * @param {Instance} filterRootInstance
   */
  function CanvasBlock( display, renderer, transformRootInstance, filterRootInstance ) {
    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  }

  scenery.register( 'CanvasBlock', CanvasBlock );

  inherit( FittedBlock, CanvasBlock, {
    initialize: function( display, renderer, transformRootInstance, filterRootInstance ) {
      this.initializeFittedBlock( display, renderer, transformRootInstance, FittedBlock.COMMON_ANCESTOR );

      // @private {Instance}
      this.filterRootInstance = filterRootInstance;

      this.dirtyDrawables = cleanArray( this.dirtyDrawables );

      if ( !this.domElement ) {
        //OHTWO TODO: support tiled Canvas handling (will need to wrap then in a div, or something)
        this.canvas = document.createElement( 'canvas' );
        this.canvas.style.position = 'absolute';
        this.canvas.style.left = '0';
        this.canvas.style.top = '0';
        this.canvas.style.pointerEvents = 'none';

        // @private {number} - unique ID so that we can support rasterization with Display.foreignObjectRasterization
        this.canvasId = this.canvas.id = 'scenery-canvas' + this.id;

        // @private {CanvasRenderingContext2D}
        this.context = this.canvas.getContext( '2d' );
        this.context.save(); // We always immediately save every Canvas so we can restore/save for clipping

        // workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
        this.context.miterLimit = 20;
        this.context.miterLimit = 10;

        // @private {CanvasContextWrapper} - Tracks intermediate Canvas context state, so we don't have to send
        // unnecessary Canvas commands.
        this.wrapper = new CanvasContextWrapper( this.canvas, this.context );

        // @public {DOMElement} - TODO: Doc this properly for {Block} as a whole
        this.domElement = this.canvas;

        // {Array.<CanvasContextWrapper>} as multiple Canvases are needed to properly render opacity within the block.
        this.wrapperStack = [ this.wrapper ];
      }
      // {number} - The index into the wrapperStack array where our current Canvas (that we are drawing to) is.
      this.wrapperStackIndex = 0;

      // Maps node ID => count of how many listeners we WOULD have attached to it. We only attach at most one listener
      // to each node. We need to listen to all ancestors up to our filter root, so that we can pick up opacity changes.
      this.filterListenerCountMap = this.filterListenerCountMap || {};

      // reset any fit transforms that were applied
      Util.prepareForTransform( this.canvas, this.forceAcceleration ); // Apply CSS needed for future CSS transforms to work properly.
      Util.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

      this.canvasDrawOffset = new Vector2( 0, 0 );

      this.currentDrawable = null;

      // @private {boolean} - Whether we need to re-apply clipping to our current Canvas
      this.clipDirty = true;

      // @private {number} - How many clips should be applied (given our current "position" in the walk up/down).
      this.clipCount = 0;

      // store our backing scale so we don't have to look it up while fitting
      this.backingScale = ( renderer & Renderer.bitmaskCanvasLowResolution ) ? 1 : scenery.Util.backingScale( this.context );

      this.clipDirtyListener = this.markDirty.bind( this );
      this.opacityDirtyListener = this.markDirty.bind( this );
      this.filterRootNode = this.filterRootInstance.node;

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `initialized #${this.id}` );
      // TODO: dirty list of nodes (each should go dirty only once, easier than scanning all?)

      return this;
    },

    setSizeFullDisplay: function() {
      var size = this.display.getSize();
      this.canvas.width = size.width * this.backingScale;
      this.canvas.height = size.height * this.backingScale;
      this.canvas.style.width = size.width + 'px';
      this.canvas.style.height = size.height + 'px';
      this.wrapper.resetStyles();
      this.canvasDrawOffset.setXY( 0, 0 );
      Util.unsetTransform( this.canvas );
    },

    setSizeFitBounds: function() {
      var x = this.fitBounds.minX;
      var y = this.fitBounds.minY;
      this.canvasDrawOffset.setXY( -x, -y ); // subtract off so we have a tight fit
      //OHTWO TODO PERFORMANCE: see if we can get a speedup by putting the backing scale in our transform instead of with CSS?
      Util.setTransform( 'matrix(1,0,0,1,' + x + ',' + y + ')', this.canvas, this.forceAcceleration ); // reapply the translation as a CSS transform
      this.canvas.width = this.fitBounds.width * this.backingScale;
      this.canvas.height = this.fitBounds.height * this.backingScale;
      this.canvas.style.width = this.fitBounds.width + 'px';
      this.canvas.style.height = this.fitBounds.height + 'px';
      this.wrapper.resetStyles();
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !FittedBlock.prototype.update.call( this ) ) {
        return false;
      }

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `update #${this.id}` );
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.push();

      while ( this.dirtyDrawables.length ) {
        this.dirtyDrawables.pop().update();
      }

      // udpate the fit BEFORE drawing, since it may change our offset
      this.updateFit();

      // for now, clear everything!
      this.context.restore(); // just in case we were clipping/etc.
      this.context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
      this.context.clearRect( 0, 0, this.canvas.width, this.canvas.height ); // clear everything
      this.context.save();
      this.wrapper.resetStyles();

      //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
      //OHTWO TODO: why is "drawable !== null" check needed
      this.currentDrawable = null; // we haven't rendered a drawable this frame yet
      for ( var drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
        this.renderDrawable( drawable );
        if ( drawable === this.lastDrawable ) { break; }
      }
      if ( this.currentDrawable ) {
        this.walkDown( this.currentDrawable.instance.trail, 0 );
      }

      assert && assert( this.clipCount === 0, 'clipCount should be zero after walking back down' );

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.pop();

      return true;
    },

    /**
     * Reapplies clips to the current context. It's necessary to fully apply every clipping area for every ancestor,
     * due to how Canvas is set up. Should ideally be called when the clip is dirty.
     * @private
     *
     * This is necessary since you can't apply "nested" clipping areas naively in Canvas, but you specify one entire
     * clip area.
     *
     * @param {CanvasSelfDrawable} Drawable
     */
    applyClip: function( drawable ) {
      this.clipDirty = false;
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `Apply clip ${drawable.instance.trail.toDebugString()}` );
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.push();

      var wrapper = this.wrapperStack[ this.wrapperStackIndex ];
      var context = wrapper.context;

      // Re-set (even if no clip is needed, so we get rid of the old clip)
      context.restore();
      context.save();
      wrapper.resetStyles();

      // If 0, no clip is needed
      if ( this.clipCount ) {
        var instance = drawable.instance;
        var trail = instance.trail;

        // Inverse of what we'll be applying to the scene, to get back to the root coordinate transform
        scratchMatrix.rowMajor( this.backingScale, 0, this.canvasDrawOffset.x * this.backingScale,
                                0, this.backingScale, this.canvasDrawOffset.y * this.backingScale,
                                0, 0, 1 );
        scratchMatrix2.set( this.transformRootInstance.trail.getMatrix() ).invert();
        scratchMatrix2.multiplyMatrix( scratchMatrix ).canvasSetTransform( context );

        // Recursively apply clips and transforms
        for ( var i = 0; i < trail.length; i++ ) {
          var node = trail.nodes[ i ];
          node.getMatrix().canvasAppendTransform( context );
          if ( node.hasClipArea() ) {
            context.beginPath();
            node.clipArea.writeToContext( context );
            // TODO: add the ability to show clipping highlights inline?
                // context.save();
                // context.strokeStyle = 'red';
                // context.lineWidth = 2;
                // context.stroke();
                // context.restore();
            context.clip();
          }
        }
      }

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.pop();
    },

    /**
     * Walk down towards the root, popping any clip/opacity effects that were needed.
     * @private
     *
     * @param {Trail} trail
     * @param {number} branchIndex - The first index where our before and after trails have diverged.
     */
    walkDown: function( trail, branchIndex ) {
      var filterRootIndex = this.filterRootInstance.trail.length - 1;

      for ( var i = trail.length - 1; i >= branchIndex; i-- ) {
        var node = trail.nodes[ i ];

        if ( node.hasClipArea() ) {
          sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `Pop clip ${trail.subtrailTo( node ).toDebugString()}` );
          // Pop clip
          this.clipCount--;
          this.clipDirty = true;
        }
        // We should not apply opacity at or below the filter root
        if ( i > filterRootIndex && node.getOpacity() !== 1 ) {
          sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `Pop opacity ${trail.subtrailTo( node ).toDebugString()}` );
          // Pop opacity
          var topWrapper = this.wrapperStack[ this.wrapperStackIndex ];
          var bottomWrapper = this.wrapperStack[ this.wrapperStackIndex - 1 ];
          this.wrapperStackIndex--;
          this.clipDirty = true;

          // Draw the transparent content into the next-level Canvas.
          bottomWrapper.context.setTransform( 1, 0, 0, 1, 0, 0 );
          bottomWrapper.context.globalAlpha = node.getOpacity();
          bottomWrapper.context.drawImage( topWrapper.canvas, 0, 0 );
          bottomWrapper.context.globalAlpha = 1;
        }
      }
    },

    /**
     * Walk up towards the next leaf, pushing any clip/opacity effects that are needed.
     * @private
     *
     * @param {Trail} trail
     * @param {number} branchIndex - The first index where our before and after trails have diverged.
     */
    walkUp: function( trail, branchIndex ) {
      var filterRootIndex = this.filterRootInstance.trail.length - 1;

      for ( var i = branchIndex; i < trail.length; i++ ) {
        var node = trail.nodes[ i ];

        // We should not apply opacity at or below the filter root
        if ( i > filterRootIndex && node.getOpacity() !== 1 ) {
          sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `Push opacity ${trail.subtrailTo( node ).toDebugString()}` );
          // Push opacity
          this.wrapperStackIndex++;
          this.clipDirty = true;
          // If we need to push an entirely new Canvas to the stack
          if ( this.wrapperStackIndex === this.wrapperStack.length ) {
            var newCanvas = document.createElement( 'canvas' );
            var newContext = newCanvas.getContext( '2d' );
            newContext.save();
            this.wrapperStack.push( new CanvasContextWrapper( newCanvas, newContext ) );
          }
          var wrapper = this.wrapperStack[ this.wrapperStackIndex ];
          var context = wrapper.context;

          // Size and clear our context
          wrapper.setDimensions( this.canvas.width, this.canvas.height );
          context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
          context.clearRect( 0, 0, this.canvas.width, this.canvas.height ); // clear everything

        }

        if ( node.hasClipArea() ) {
          sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `Push clip ${trail.subtrailTo( node ).toDebugString()}` );
          // Push clip
          this.clipCount++;
          this.clipDirty = true;
        }
      }
    },

    /**
     * Draws the drawable into our main Canvas.
     * @private
     *
     * For things like opacity/clipping, as part of this we walk up/down part of the instance tree for rendering each
     * drawable.
     *
     * @param {CanvasSelfDrawable} - TODO: In the future, we'll need to support Canvas caches (this should be updated
     *                               with a proper generalized type)
     */
    renderDrawable: function( drawable ) {
      // do not paint invisible drawables
      if ( !drawable.visible ) {
        return;
      }

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `renderDrawable #${drawable.id} ${drawable.instance.trail.toDebugString()}`  );
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.push();

      // For opacity/clip, walk up/down as necessary (Can only walk down if we are not the first drawable)
      var branchIndex = this.currentDrawable ? drawable.instance.getBranchIndexTo( this.currentDrawable.instance ) : 0;
      if ( this.currentDrawable ) {
        this.walkDown( this.currentDrawable.instance.trail, branchIndex );
      }
      this.walkUp( drawable.instance.trail, branchIndex );

      var wrapper = this.wrapperStack[ this.wrapperStackIndex ];
      var context = wrapper.context;

      // Re-apply the clip if necessary. The walk down/up may have flagged a potential clip change (if we walked across
      // something with a clip area).
      if ( this.clipDirty ) {
        this.applyClip( drawable );
      }

      // we're directly accessing the relative transform below, so we need to ensure that it is up-to-date
      assert && assert( drawable.instance.relativeTransform.isValidationNotNeeded() );

      var matrix = drawable.instance.relativeTransform.matrix;

      // set the correct (relative to the transform root) transform up, instead of walking the hierarchy (for now)
      //OHTWO TODO: should we start premultiplying these matrices to remove this bottleneck?
      context.setTransform( this.backingScale, 0, 0, this.backingScale, this.canvasDrawOffset.x * this.backingScale, this.canvasDrawOffset.y * this.backingScale );
      if ( drawable.instance !== this.transformRootInstance ) {
        matrix.canvasAppendTransform( context );
      }

      // paint using its local coordinate frame
      drawable.paintCanvas( wrapper, drawable.instance.node, drawable.instance.relativeTransform.matrix );

      this.currentDrawable = drawable;

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.pop();
    },

    dispose: function() {
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `dispose #${this.id}` );

      this.filterRootNode = null;

      // clear references
      this.transformRootInstance = null;
      cleanArray( this.dirtyDrawables );

      // minimize memory exposure of the backing raster
      this.canvas.width = 0;
      this.canvas.height = 0;

      FittedBlock.prototype.dispose.call( this );
    },

    markDirtyDrawable: function( drawable ) {
      sceneryLog && sceneryLog.dirty && sceneryLog.dirty( `markDirtyDrawable on CanvasBlock#${this.id} with ${drawable.toString()}` );

      assert && assert( drawable );

      if ( assert ) {
        // Catch infinite loops
        this.display.ensureNotPainting();
      }

      // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
      this.dirtyDrawables.push( drawable );
      this.markDirty();
    },

    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `#${this.id}.addDrawable ${drawable.toString()}` );

      FittedBlock.prototype.addDrawable.call( this, drawable );

      // Add opacity listeners (from this node up to the filter root)
      for ( var instance = drawable.instance; instance && instance !== this.filterRootInstance; instance = instance.parent ) {
        var node = instance.node;

        // Only add the listener if we don't already have one
        if ( this.filterListenerCountMap[ node.id ] ) {
          this.filterListenerCountMap[ node.id ]++;
        }
        else {
          this.filterListenerCountMap[ node.id ] = 1;

          node.onStatic( 'opacity', this.opacityDirtyListener );
          node.onStatic( 'clip', this.clipDirtyListener );
        }
      }
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `#${this.id}.removeDrawable ${drawable.toString()}` );

      // Remove opacity listeners (from this node up to the filter root)
      for ( var instance = drawable.instance; instance && instance !== this.filterRootInstance; instance = instance.parent ) {
        var node = instance.node;
        assert && assert( this.filterListenerCountMap[ node.id ] > 0 );
        this.filterListenerCountMap[ node.id ]--;
        if ( this.filterListenerCountMap[ node.id ] === 0 ) {
          delete this.filterListenerCountMap[ node.id ];

          node.offStatic( 'clip', this.clipDirtyListener );
          node.offStatic( 'opacity', this.opacityDirtyListener );
        }
      }

      FittedBlock.prototype.removeDrawable.call( this, drawable );
    },

    onIntervalChange: function( firstDrawable, lastDrawable ) {
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `#${this.id}.onIntervalChange ${firstDrawable.toString()} to ${lastDrawable.toString()}` );

      FittedBlock.prototype.onIntervalChange.call( this, firstDrawable, lastDrawable );

      // If we have an interval change, we'll need to ensure we repaint (even if we're full-display). This was a missed
      // case for https://github.com/phetsims/scenery/issues/512, where it would only clear if it was a common-ancestor
      // fitted block.
      this.markDirty();
    },

    onPotentiallyMovedDrawable: function( drawable ) {
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.CanvasBlock( `#${this.id}.onPotentiallyMovedDrawable ${drawable.toString()}` );
      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.push();

      assert && assert( drawable.parentDrawable === this );

      // For now, mark it as dirty so that we redraw anything containing it. In the future, we could have more advanced
      // behavior that figures out the intersection-region for what was moved and what it was moved past, but that's
      // a harder problem.
      drawable.markDirty();

      sceneryLog && sceneryLog.CanvasBlock && sceneryLog.pop();
    },

    toString: function() {
      return `CanvasBlock#${this.id}-${FittedBlock.fitString[ this.fit ]}`;
    }
  } );

  Poolable.mixInto( CanvasBlock, {
    initialize: CanvasBlock.prototype.initialize
  } );

  return CanvasBlock;
} );
