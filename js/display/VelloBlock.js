// Copyright 2023, University of Colorado Boulder

/**
 * Renders a visual layer of WebGL drawables.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Sharfudeen Ashraf (For Ghent University)
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import cleanArray from '../../../phet-core/js/cleanArray.js';
import Poolable from '../../../phet-core/js/Poolable.js';
import { Compose, DeviceContext, FilterMatrix, FittedBlock, Mix, PhetEncoding, render, scenery, Utils } from '../imports.js';

const scalingMatrix = Matrix3.scaling( window.devicePixelRatio );
const scratchFilterMatrix = new FilterMatrix();

class VelloBlock extends FittedBlock {
  /**
   * @mixes Poolable
   *
   * @param {Display} display
   * @param {number} renderer
   * @param {Instance} transformRootInstance
   * @param {Instance} filterRootInstance
   */
  constructor( display, renderer, transformRootInstance, filterRootInstance ) {
    super();

    this.initialize( display, renderer, transformRootInstance, filterRootInstance );
  }

  /**
   * @public
   *
   * @param {Display} display
   * @param {number} renderer
   * @param {Instance} transformRootInstance
   * @param {Instance} filterRootInstance
   * @returns {VelloBlock} - For chaining
   */
  initialize( display, renderer, transformRootInstance, filterRootInstance ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `initialize #${this.id}` );
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.push();

    // VelloBlocks are hard-coded to take the full display size (as opposed to svg and canvas)
    // Since we saw some jitter on iPad, see #318 and generally expect WebGPU layers to span the entire display
    // In the future, it would be good to understand what was causing the problem and make webgl consistent
    // with svg and canvas again.
    // TODO: Don't have it be a "Fitted" block then
    super.initialize( display, renderer, transformRootInstance, FittedBlock.FULL_DISPLAY );

    this.filterRootInstance = filterRootInstance;

    // // {boolean} - Whether we pass this flag to the WebGL Context. It will store the contents displayed on the screen,
    // // so that canvas.toDataURL() will work. It also requires clearing the context manually ever frame. Both incur
    // // performance costs, so it should be false by default.
    // // TODO: This block can be shared across displays, so we need to handle preserveDrawingBuffer separately?
    // this.preserveDrawingBuffer = display._preserveDrawingBuffer;

    // list of {Drawable}s that need to be updated before we update
    this.dirtyDrawables = cleanArray( this.dirtyDrawables );

    // @private {Object.<nodeId:number,number> - Maps node ID => count of how many listeners we WOULD have attached to
    // it. We only attach at most one listener to each node. We need to listen to all ancestors up to our filter root,
    // so that we can pick up opacity changes.
    this.filterListenerCountMap = this.filterListenerCountMap || {};

    // @private {function}
    this.clipDirtyListener = this.clipDirtyListener || this.markDirty.bind( this );
    this.opacityDirtyListener = this.opacityDirtyListener || this.markDirty.bind( this );

    // @private {Drawable|null} -- The drawable for which the current clip/filter setup has been applied (during the walk)
    this.currentDrawable = null;

    if ( !this.domElement ) {
      // @public (scenery-internal) {HTMLCanvasElement} - Div wrapper used so we can switch out Canvases if necessary.
      this.canvas = document.createElement( 'canvas' );
      this.canvas.style.position = 'absolute';
      this.canvas.style.left = '0';
      this.canvas.style.top = '0';
      this.canvas.style.pointerEvents = 'none';

      // @private {number} - unique ID so that we can support rasterization with Display.foreignObjectRasterization
      this.canvasId = this.canvas.id = `scenery-vello${this.id}`;

      this.domElement = this.canvas;

      // TODO: handle context restoration/loss
      this.deviceContext = DeviceContext.getSync();
      this.canvasContext = this.deviceContext.getCanvasContext( this.canvas );
    }

    // reset any fit transforms that were applied
    Utils.prepareForTransform( this.canvas ); // Apply CSS needed for future CSS transforms to work properly.
    Utils.unsetTransform( this.canvas ); // clear out any transforms that could have been previously applied

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.pop();

    return this;
  }

  static VELLO_BLOCK_APPLIES_OPACITY_FILTERS = true;

  /**
   * @public
   * @override
   */
  setSizeFullDisplay() {
    const size = this.display.getSize();
    this.canvas.width = Math.ceil( size.width * window.devicePixelRatio );
    this.canvas.height = Math.ceil( size.height * window.devicePixelRatio );
    this.canvas.style.width = `${size.width}px`;
    this.canvas.style.height = `${size.height}px`;
  }

  /**
   * @public
   * @override
   */
  setSizeFitBounds() {
    throw new Error( 'setSizeFitBounds unimplemented for VelloBlock' );
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   * @override
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
    if ( !super.update() ) {
      return false;
    }

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `update #${this.id}` );
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.push();

    // update drawables, so that they have vertex arrays up to date, etc.
    while ( this.dirtyDrawables.length ) {
      this.dirtyDrawables.pop().update();
    }

    // update the fit BEFORE drawing, since it may change our offset
    // TODO: make not fittable
    this.updateFit();

    const encoding = new PhetEncoding();
    encoding.reset( false );

    // Iterate through all of our drawables (linked list)
    //OHTWO TODO: PERFORMANCE: create an array for faster drawable iteration (this is probably a hellish memory access pattern)
    this.currentDrawable = null; // we haven't rendered a drawable this frame yet
    for ( let drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
      // ignore invisible drawables
      if ( drawable.visible ) {
        // For opacity/clip, walk up/down as necessary (Can only walk down if we are not the first drawable)
        const branchIndex = this.currentDrawable ? drawable.instance.getBranchIndexTo( this.currentDrawable.instance ) : 0;
        if ( this.currentDrawable ) {
          this.walkDown( encoding, this.currentDrawable.instance.trail, branchIndex );
        }
        this.walkUp( encoding, drawable.instance.trail, branchIndex );

        encoding.append( drawable.encoding );

        this.currentDrawable = drawable;
      }

      // exit loop end case
      if ( drawable === this.lastDrawable ) { break; }
    }
    if ( this.currentDrawable ) {
      this.walkDown( encoding, this.currentDrawable.instance.trail, 0 );
    }

    if ( sceneryLog && sceneryLog.Encoding ) {

      this.lastEncodingString = encoding.rustEncoding +
                                `sb.append(&SceneFragment { data: encoding${encoding.id} }, None);\n`;
    }

    encoding.finalizeScene();

    const outTexture = this.canvasContext.getCurrentTexture();
    const renderInfo = encoding.resolve( this.deviceContext );
    renderInfo.prepareRender( outTexture.width, outTexture.height, 0, false );

    render( renderInfo, this.deviceContext, outTexture );

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.pop();

    return true;
  }

  /**
   * TODO: code share with Canvas, somehow perhaps?
   * Walk down towards the root, popping any clip/opacity effects that were needed.
   * @private
   *
   * @param {PhetEncoding} encoding
   * @param {Trail} trail
   * @param {number} branchIndex - The first index where our before and after trails have diverged.
   */
  walkDown( encoding, trail, branchIndex ) {
    if ( !VelloBlock.VELLO_BLOCK_APPLIES_OPACITY_FILTERS ) {
      return;
    }

    const filterRootIndex = this.filterRootInstance.trail.length - 1;

    for ( let i = trail.length - 1; i >= branchIndex; i-- ) {
      const node = trail.nodes[ i ];

      let needsEncodeEndClip = false;

      if ( node.hasClipArea() ) {
        sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `Pop clip ${trail.subtrailTo( node ).toDebugString()}` );

        needsEncodeEndClip = true;
      }

      // We should not apply opacity or other filters at or below the filter root
      if ( i > filterRootIndex ) {
        if ( node._filters.length ) {
          sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `Pop filters ${trail.subtrailTo( node ).toDebugString()}` );

          for ( let i = 0; i < node._filters.length; i++ ) {
            const filter = node._filters[ i ];
            if ( filter.isVelloCompatible() ) {
              needsEncodeEndClip = true;
            }
          }
        }

        if ( node.getEffectiveOpacity() !== 1 ) {
          sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `Pop opacity ${trail.subtrailTo( node ).toDebugString()}` );

          needsEncodeEndClip = true;
        }
      }

      if ( needsEncodeEndClip ) {
        encoding.encodeEndClip();
      }
    }
  }

  /**
   * Walk up towards the next leaf, pushing any clip/opacity effects that are needed.
   * TODO: code share with Canvas?
   * @private
   *
   * @param {PhetEncoding} encoding
   * @param {Trail} trail
   * @param {number} branchIndex - The first index where our before and after trails have diverged.
   */
  walkUp( encoding, trail, branchIndex ) {
    if ( !VelloBlock.VELLO_BLOCK_APPLIES_OPACITY_FILTERS ) {
      return;
    }

    const filterRootIndex = this.filterRootInstance.trail.length - 1;

    for ( let i = branchIndex; i < trail.length; i++ ) {
      const node = trail.nodes[ i ];

      let needsEncodeBeginClip = false;
      let alpha = 1;
      let clipArea = null;

      const filterMatrix = scratchFilterMatrix;
      filterMatrix.reset();

      // We should not apply opacity at or below the filter root
      if ( i > filterRootIndex ) {
        alpha = node.getEffectiveOpacity();
        if ( alpha !== 1 ) {
          sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `Push opacity ${trail.subtrailTo( node ).toDebugString()}` );

          needsEncodeBeginClip = true;

          filterMatrix.multiplyAlpha( alpha );
        }

        if ( node._filters.length ) {
          sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `Push filters ${trail.subtrailTo( node ).toDebugString()}` );

          for ( let j = 0; j < node._filters.length; j++ ) {
            const filter = node._filters[ j ];
            if ( filter.isVelloCompatible() ) {
              needsEncodeBeginClip = true;

              // NOTE: Assumes ColorMatrixFilter
              filterMatrix.multiply( filter );
            }
          }
        }
      }

      if ( node.hasClipArea() && node.getClipArea().getNonoverlappingArea() > 0 ) {
        sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `Push clip ${trail.subtrailTo( node ).toDebugString()}` );

        needsEncodeBeginClip = true;
        clipArea = node.getClipArea();
      }

      if ( needsEncodeBeginClip ) {
        // For a layer push: matrix, linewidth(-1), shape, begin_clip

        if ( clipArea ) {
          // +1 ideally to avoid including the filter root (ignore its parent coordinate frame, stay in its local)
          encoding.encodeMatrix( scalingMatrix.timesMatrix( trail.slice( this.transformRootInstance.trail.length, i + 1 ).getMatrix() ) );

        }
        else {
          encoding.encodeMatrix( Matrix3.IDENTITY );
        }

        encoding.encodeLineWidth( -1 );

        if ( clipArea ) {
          // TODO: consolidate tolerance somewhere. Adaptively set this up? ACTUALLY we should really avoid
          // TODO: re-encoding the clips like this every frame, right?
          encoding.encodeShape( clipArea, true, true, 0.01 );
        }
        else {
          encoding.encodeRect( 0, 0, this.canvas.width, this.canvas.height );
        }

        // TODO: filters we can do with this
        // TODO: ensure NOT Mix.Clip when alpha < 1 (but we can gain performance if alpha is 1?)
        encoding.encodeBeginClip( Mix.Normal, Compose.SrcOver, filterMatrix );
      }
    }
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `dispose #${this.id}` );

    // TODO: many things to dispose!?

    // clear references
    cleanArray( this.dirtyDrawables );

    super.dispose();
  }

  /**
   * @public
   *
   * @param {Drawable} drawable
   */
  markDirtyDrawable( drawable ) {
    sceneryLog && sceneryLog.dirty && sceneryLog.dirty( `markDirtyDrawable on VelloBlock#${this.id} with ${drawable.toString()}` );

    assert && assert( drawable );
    assert && assert( !drawable.isDisposed );

    // TODO: instance check to see if it is a canvas cache (usually we don't need to call update on our drawables)
    this.dirtyDrawables.push( drawable );
    this.markDirty();
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  addDrawable( drawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.addDrawable ${drawable.toString()}` );

    super.addDrawable( drawable );

    // Add opacity listeners (from this node up to the filter root)
    for ( let instance = drawable.instance; instance && instance !== this.filterRootInstance; instance = instance.parent ) {
      const node = instance.node;

      // Only add the listener if we don't already have one
      if ( this.filterListenerCountMap[ node.id ] ) {
        this.filterListenerCountMap[ node.id ]++;
      }
      else {
        this.filterListenerCountMap[ node.id ] = 1;

        node.filterChangeEmitter.addListener( this.opacityDirtyListener );
        node.clipAreaProperty.lazyLink( this.clipDirtyListener );
      }
    }
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} drawable
   */
  removeDrawable( drawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.removeDrawable ${drawable.toString()}` );

    // Remove opacity listeners (from this node up to the filter root)
    for ( let instance = drawable.instance; instance && instance !== this.filterRootInstance; instance = instance.parent ) {
      const node = instance.node;
      assert && assert( this.filterListenerCountMap[ node.id ] > 0 );
      this.filterListenerCountMap[ node.id ]--;
      if ( this.filterListenerCountMap[ node.id ] === 0 ) {
        delete this.filterListenerCountMap[ node.id ];

        node.clipAreaProperty.unlink( this.clipDirtyListener );
        node.filterChangeEmitter.removeListener( this.opacityDirtyListener );
      }
    }

    // Ensure a removed drawable is not present in the dirtyDrawables array afterwards. Don't want to update it.
    // See https://github.com/phetsims/scenery/issues/635
    let index = 0;
    while ( ( index = this.dirtyDrawables.indexOf( drawable, index ) ) >= 0 ) {
      this.dirtyDrawables.splice( index, 1 );
    }

    super.removeDrawable( drawable );
  }

  /**
   * @public
   * @override
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  onIntervalChange( firstDrawable, lastDrawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.onIntervalChange ${firstDrawable.toString()} to ${lastDrawable.toString()}` );

    super.onIntervalChange( firstDrawable, lastDrawable );

    this.markDirty();
  }

  /**
   * @public
   *
   * @param {Drawable} drawable
   */
  onPotentiallyMovedDrawable( drawable ) {
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.VelloBlock( `#${this.id}.onPotentiallyMovedDrawable ${drawable.toString()}` );
    sceneryLog && sceneryLog.VelloBlock && sceneryLog.push();

    assert && assert( drawable.parentDrawable === this );

    this.markDirty();

    sceneryLog && sceneryLog.VelloBlock && sceneryLog.pop();
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return `VelloBlock#${this.id}-${FittedBlock.fitString[ this.fit ]}`;
  }
}

scenery.register( 'VelloBlock', VelloBlock );

Poolable.mixInto( VelloBlock );

export default VelloBlock;