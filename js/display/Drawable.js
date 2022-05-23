// Copyright 2013-2022, University of Colorado Boulder

/**
 * Something that can be displayed with a specific renderer.
 * NOTE: Drawables are assumed to be pooled with PoolableMixin, as freeToPool() is called.
 *
 * A drawable's life-cycle starts with its initialization (calling initialize once), and ends with its disposal
 * (where it is freed to its own pool).
 *
 * Drawables are part of an unordered drawable "tree" where each drawable can have a parent references. This is used
 * for, among other things, propagation of 'dirty' flags and usage during stitching.
 *
 * Blocks and backbones (sub-types of Drawable) contain children (creating a tree, although shared caches make it more
 * like a DAG). Our Scenery Display is built from a root backbone, that contains blocks. This can be Canvas/SVG, but
 * may also contain a DOM block with another backbone (used for opacity, CSS transforms, etc.).
 *
 * Drawables are part of two inherent linked lists: an "old" and a "new" one. Usually they are the same, but during
 * updates, the "new" linked list is changed to accomodate any changes, and then a stitch process is done to mark which
 * block (parent) we will belong to.
 *
 * As part of stitching or other processes, a Drawable is responsible for recording its pending state changes. Most
 * notably, we need to determine whether a drawable is being added, moved, or removed in the next frame. This is done
 * with an idempotent API using notePendingAddition/notePendingRemoval/notePendingMove. Either:
 *   - One or more notePendingMove() calls are made. When we are updated with updateBlock(), we will move to the
 *     last block referenced with notePendingMove() (which may be a no-op if it is the same block).
 *   - Zero or one notePendingAddition() call is made, and zero or one notePendingRemoval() call is made. Our action is:
 *     - No addition, no removal: nothing done
 *     - No addition, one removal: We are removed from our last block (and then presumably disposed later)
 *     - One addition, no removal: We are added to our new (pending) block, without being removed from anything
 *     - One addition, one removal: We are removed from our last block and added to our new (pending) block.
 * It is set up so that the order of addition/removal calls doesn't matter, since these can occur from within different
 * backbone stitches (removed in one, added in another, or with the order reversed). Our updateBlocks() is guaranteed
 * to be called after all of those have been completed.
 *
 * APIs for drawable types:
 *
 * DOM: {
 *   domElement: {HTMLElement}
 * }
 * Canvas: {
 *   paintCanvas: function( {CanvasContextWrapper} wrapper, {Node} node, {Matrix3} matrix )
 * }
 * SVG: {
 *   svgElement: {SVGElement}
 * }
 * WebGL: {
 *   onAddToBlock: function( {WebGLBlock} block )
 *   onRemoveFromBlock: function( {WebGLBlock} block )
 *   render: function( {ShaderProgram} shaderProgram )
 *   shaderAttributes: {string[]} - names of vertex attributes to be used
 * }
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import { Block, Renderer, scenery } from '../imports.js';

let globalId = 1;

class Drawable {
  /**
   * @public
   *
   * @param {number} renderer
   * @returns {Drawable} - for chaining
   */
  initialize( renderer ) {

    assert && assert( !this.id || this.isDisposed, 'If we previously existed, we need to have been disposed' );

    // @public {number} - unique ID for drawables
    this.id = this.id || globalId++;

    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] initialize ${this.toString()}` );

    this.clean();

    // @public {number} - Bitmask defined by Renderer.js
    this.renderer = renderer;

    // @public {boolean}
    this.dirty = true;

    // @private {boolean}
    this.isDisposed = false;
    this.linksDirty = false;

    // @public {TinyProperty.<boolean>}
    this.visibleProperty = new TinyProperty( true );
    this.fittableProperty = new TinyProperty( true ); // If false, will cause our parent block to not be fitted

    return this;
  }

  /**
   * Cleans the state of this drawable to the defaults.
   * @protected
   */
  clean() {
    // @public {Drawable|null} - what drawable we are being rendered (or put) into (will be filled in later)
    this.parentDrawable = null;

    // @public {BackboneDrawable|null} - a backbone reference (if applicable).
    this.backbone = null;

    // @public {Drawable|null} - what our parent drawable will be after the stitch is finished
    this.pendingParentDrawable = null;

    // @public {BackboneDrawable|null} - what our backbone will be after the stitch is finished (if applicable)
    this.pendingBackbone = null;

    // @public {boolean} - whether we are to be added to a block/backbone in our updateBlock() call
    this.pendingAddition = false;

    // @public {boolean} - whether we are to be removed from a block/backbone in our updateBlock() call
    this.pendingRemoval = false;

    assert && assert( !this.previousDrawable && !this.nextDrawable,
      'By cleaning (disposal or fresh creation), we should have disconnected from the linked list' );

    // @public {Drawable|null} - Linked list handling (will be filled in later)
    this.previousDrawable = null;
    this.nextDrawable = null;

    // @public {Drawable|null} - Similar to previousDrawable/nextDrawable, but without recent changes, so that we can
    // traverse both orders at the same time for stitching.
    this.oldPreviousDrawable = null;
    this.oldNextDrawable = null;

    this.visibleProperty && this.visibleProperty.removeAllListeners();
    this.fittableProperty && this.fittableProperty.removeAllListeners();
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   *
   * Generally meant to be overridden in subtypes (but should still call this to check if they should update).
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    let needsFurtherUpdates = false;

    if ( this.dirty && !this.isDisposed ) {
      this.dirty = false;
      needsFurtherUpdates = true;
    }

    return needsFurtherUpdates;
  }

  /**
   * Sets whether the drawable is visible.
   * @public
   *
   * @param {boolean} visible
   */
  setVisible( visible ) {
    this.visibleProperty.value = visible;
  }

  set visible( value ) { this.setVisible( value ); }

  /**
   * Returns whether the drawable is visible.
   * @public
   *
   * @returns {boolean}
   */
  isVisible() {
    return this.visibleProperty.value;
  }

  get visible() { return this.isVisible(); }

  /**
   * Sets whether this drawable is fittable.
   * @public
   *
   * NOTE: Should be called just after initialization (before being added to blocks) if we aren't fittable.
   *
   * @param {boolean} fittable
   */
  setFittable( fittable ) {
    this.fittableProperty.value = fittable;
  }

  set fittable( value ) { this.setFittable( value ); }

  /**
   * Returns whether the drawable is fittable.
   * @public
   *
   * @returns {boolean}
   */
  isFittable() {
    return this.fittableProperty.value;
  }

  get fittable() { return this.isFittable(); }

  /**
   * Called to add a block (us) as a child of a backbone
   * @public
   *
   * @param {BackboneDrawable} backboneInstance
   */
  setBlockBackbone( backboneInstance ) {
    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] setBlockBackbone ${
      this.toString()} with ${backboneInstance.toString()}` );

    // if this is being called, Block will be guaranteed to be loaded
    assert && assert( this instanceof Block );

    this.parentDrawable = backboneInstance;
    this.backbone = backboneInstance;
    this.pendingParentDrawable = backboneInstance;
    this.pendingBackbone = backboneInstance;
    this.pendingAddition = false;
    this.pendingRemoval = false;
  }

  /**
   * Notifies the Display of a pending addition.
   * @public
   *
   * @param {Display} display
   * @param {Block} block
   * @param {BackboneDrawable} backbone
   */
  notePendingAddition( display, block, backbone ) {
    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] notePendingAddition ${
      this.toString()} with ${block.toString()}, ${
      backbone ? backbone.toString() : '-'}` );

    assert && assert( backbone !== undefined, 'backbone can be either null or a backbone' );
    assert && assert( block instanceof Block );

    this.pendingParentDrawable = block;
    this.pendingBackbone = backbone;
    this.pendingAddition = true;

    // if we weren't already marked for an update, mark us
    if ( !this.pendingRemoval ) {
      display.markDrawableChangedBlock( this );
    }
  }

  /**
   * Notifies the Display of a pending removal.
   * @public
   *
   * @param {Display} display
   */
  notePendingRemoval( display ) {
    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] notePendingRemoval ${
      this.toString()}` );

    this.pendingRemoval = true;

    // if we weren't already marked for an update, mark us
    if ( !this.pendingAddition ) {
      display.markDrawableChangedBlock( this );
    }
  }

  /**
   * Notifies the Display of a pending move.
   * @public
   *
   * Moving a drawable that isn't changing backbones, just potentially changing its block.
   * It should not have notePendingAddition or notePendingRemoval called on it.
   *
   * @param {Display} display
   * @param {Block} block
   */
  notePendingMove( display, block ) {
    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] notePendingMove ${
      this.toString()} with ${block.toString()}` );

    assert && assert( block instanceof Block );

    this.pendingParentDrawable = block;

    if ( !this.pendingRemoval || !this.pendingAddition ) {
      display.markDrawableChangedBlock( this );
    }

    // set both flags, since we need it to be removed and added
    this.pendingAddition = true;
    this.pendingRemoval = true;
  }

  /**
   * Updates the block.
   * @public
   *
   * @returns {boolean} - Whether we changed our block
   */
  updateBlock() {
    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] updateBlock ${this.toString()
    } with add:${this.pendingAddition
    } remove:${this.pendingRemoval
    } old:${this.parentDrawable ? this.parentDrawable.toString() : '-'
    } new:${this.pendingParentDrawable ? this.pendingParentDrawable.toString() : '-'}` );
    sceneryLog && sceneryLog.Drawable && sceneryLog.push();

    let changed = false;

    if ( this.pendingRemoval || this.pendingAddition ) {
      // we are only unchanged if we have an addition AND removal, and the endpoints are identical
      changed = !this.pendingRemoval || !this.pendingAddition ||
                this.parentDrawable !== this.pendingParentDrawable ||
                this.backbone !== this.pendingBackbone;

      if ( changed ) {
        if ( this.pendingRemoval ) {
          sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `removing from ${this.parentDrawable.toString()}` );
          this.parentDrawable.removeDrawable( this );

          // remove references if we are not being added back in
          if ( !this.pendingAddition ) {
            this.pendingParentDrawable = null;
            this.pendingBackbone = null;
          }
        }

        this.parentDrawable = this.pendingParentDrawable;
        this.backbone = this.pendingBackbone;

        if ( this.pendingAddition ) {
          sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `adding to ${this.parentDrawable.toString()}` );
          this.parentDrawable.addDrawable( this );
        }
      }
      else {
        sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( 'unchanged' );

        if ( this.pendingAddition && Renderer.isCanvas( this.renderer ) ) {
          this.parentDrawable.onPotentiallyMovedDrawable( this );
        }
      }

      this.pendingAddition = false;
      this.pendingRemoval = false;
    }

    sceneryLog && sceneryLog.Drawable && sceneryLog.pop();

    return changed;
  }

  /**
   * Moves the old-drawable-linked-list information into the current-linked-list.
   * @public
   */
  updateLinks() {
    this.oldNextDrawable = this.nextDrawable;
    this.oldPreviousDrawable = this.previousDrawable;
    this.linksDirty = false;
  }

  /**
   * Marks this as needing an update.
   * @public
   */
  markDirty() {
    if ( !this.dirty ) {
      this.dirty = true;

      // TODO: notify what we want to call repaint() later
      if ( this.parentDrawable ) {
        this.parentDrawable.markDirtyDrawable( this );
      }
    }
  }

  /**
   * Marks our linked list as dirty.
   * @public
   *
   * Will ensure that after syncTree phase is done, we will have updateLinks() called on us
   *
   * @param {Display} display
   */
  markLinksDirty( display ) {
    if ( !this.linksDirty ) {
      this.linksDirty = true;
      display.markDrawableForLinksUpdate( this );
    }
  }

  /**
   * Marks us for disposal in the next phase of updateDisplay(), and disconnects from the linked list
   * @public
   *
   * @param {Display} display
   */
  markForDisposal( display ) {
    // as we are marked for disposal, we disconnect from the linked list (so our disposal setting nulls won't cause issues)
    Drawable.disconnectBefore( this, display );
    Drawable.disconnectAfter( this, display );

    display.markDrawableForDisposal( this );
  }

  /**
   * Disposes immediately, and makes no guarantees about out linked list's state (disconnects).
   * @public
   *
   * @param {Display} display
   */
  disposeImmediately( display ) {
    // as we are marked for disposal, we disconnect from the linked list (so our disposal setting nulls won't cause issues)
    Drawable.disconnectBefore( this, display );
    Drawable.disconnectAfter( this, display );

    this.dispose();
  }

  /**
   * Releases references
   * @public
   *
   * NOTE: Generally do not call this directly, use markForDisposal (so Display will dispose us), or disposeImmediately.
   *
   * @param {*} !this.isDisposed
   * @param {*} 'We should not re-dispose drawables'
   */
  dispose() {
    assert && assert( !this.isDisposed, 'We should not re-dispose drawables' );

    sceneryLog && sceneryLog.Drawable && sceneryLog.Drawable( `[${this.constructor.name}*] dispose ${this.toString()}` );
    sceneryLog && sceneryLog.Drawable && sceneryLog.push();

    this.clean();
    this.isDisposed = true;

    // for now
    this.freeToPool();

    sceneryLog && sceneryLog.Drawable && sceneryLog.pop();
  }

  /**
   * Runs checks on the drawable, based on certain flags.
   * @public
   *
   * @param {boolean} allowPendingBlock
   * @param {boolean} allowPendingList
   * @param {boolean} allowDirty
   */
  audit( allowPendingBlock, allowPendingList, allowDirty ) {
    if ( assertSlow ) {
      assertSlow && assertSlow( !this.isDisposed,
        'If we are being audited, we assume we are in the drawable display tree, and we should not be marked as disposed' );
      assertSlow && assertSlow( this.renderer, 'Should not have a 0 (no) renderer' );

      assertSlow && assertSlow( !this.backbone || this.parentDrawable,
        'If we have a backbone reference, we must have a parentDrawable (our block)' );

      if ( !allowPendingBlock ) {
        assertSlow && assertSlow( !this.pendingAddition );
        assertSlow && assertSlow( !this.pendingRemoval );
        assertSlow && assertSlow( this.parentDrawable === this.pendingParentDrawable,
          'Assure our parent and pending parent match, if we have updated blocks' );
        assertSlow && assertSlow( this.backbone === this.pendingBackbone,
          'Assure our backbone and pending backbone match, if we have updated blocks' );
      }

      if ( !allowPendingList ) {
        assertSlow && assertSlow( this.oldPreviousDrawable === this.previousDrawable,
          'Pending linked-list references should be cleared by now' );
        assertSlow && assertSlow( this.oldNextDrawable === this.nextDrawable,
          'Pending linked-list references should be cleared by now' );
        assertSlow && assertSlow( !this.linksDirty, 'Links dirty flag should be clean' );
      }

      if ( !allowDirty ) {
        assertSlow && assertSlow( !this.dirty,
          'Should not be dirty at this phase, if we are in the drawable display tree' );
      }
    }
  }

  /**
   * Returns a string form of this object
   * @public
   *
   * @returns {string}
   */
  toString() {
    return `${this.constructor.name}#${this.id}`;
  }

  /**
   * Returns a more-informative string form of this object.
   * @public
   *
   * @returns {string}
   */
  toDetailedString() {
    return this.toString();
  }

  /**
   * Connects the two drawables in the linked list, while cutting the previous connection and marking
   * @public
   *
   * @param {Drawable} a
   * @param {Drawable} b
   * @param {Display} display
   */
  static connectDrawables( a, b, display ) {
    // we don't need to do anything if there is no change
    if ( a.nextDrawable !== b ) {
      // touch previous neighbors
      if ( a.nextDrawable ) {
        a.nextDrawable.markLinksDirty( display );
        a.nextDrawable.previousDrawable = null;
      }
      if ( b.previousDrawable ) {
        b.previousDrawable.markLinksDirty( display );
        b.previousDrawable.nextDrawable = null;
      }

      a.nextDrawable = b;
      b.previousDrawable = a;

      // mark these as needing updates
      a.markLinksDirty( display );
      b.markLinksDirty( display );
    }
  }

  /**
   * Disconnects the previous/before drawable from the provided one (for the linked list).
   * @public
   *
   * @param {Drawable} drawable
   * @param {Display} display
   */
  static disconnectBefore( drawable, display ) {
    // we don't need to do anything if there is no change
    if ( drawable.previousDrawable ) {
      drawable.markLinksDirty( display );
      drawable.previousDrawable.markLinksDirty( display );
      drawable.previousDrawable.nextDrawable = null;
      drawable.previousDrawable = null;
    }
  }

  /**
   * Disconnects the next/after drawable from the provided one (for the linked list).
   * @public
   *
   * @param {Drawable} drawable
   * @param {Display} display
   */
  static disconnectAfter( drawable, display ) {
    // we don't need to do anything if there is no change
    if ( drawable.nextDrawable ) {
      drawable.markLinksDirty( display );
      drawable.nextDrawable.markLinksDirty( display );
      drawable.nextDrawable.previousDrawable = null;
      drawable.nextDrawable = null;
    }
  }

  /**
   * Converts a linked list of drawables to an array (useful for debugging/assertion purposes, should not be used in
   * production code).
   * @public
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   * @returns {Array.<Drawable>}
   */
  static listToArray( firstDrawable, lastDrawable ) {
    const arr = [];

    // assumes we'll hit lastDrawable, otherwise we'll NPE
    for ( let drawable = firstDrawable; ; drawable = drawable.nextDrawable ) {
      arr.push( drawable );

      if ( drawable === lastDrawable ) {
        break;
      }
    }

    return arr;
  }

  /**
   * Converts an old linked list of drawables to an array (useful for debugging/assertion purposes, should not be
   * used in production code)
   * @public
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   * @returns {Array.<Drawable>}
   */
  static oldListToArray( firstDrawable, lastDrawable ) {
    const arr = [];

    // assumes we'll hit lastDrawable, otherwise we'll NPE
    for ( let drawable = firstDrawable; ; drawable = drawable.oldNextDrawable ) {
      arr.push( drawable );

      if ( drawable === lastDrawable ) {
        break;
      }
    }

    return arr;
  }
}

scenery.register( 'Drawable', Drawable );
export default Drawable;