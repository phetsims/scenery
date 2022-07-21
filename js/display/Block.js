// Copyright 2014-2022, University of Colorado Boulder

/**
 * A specialized drawable for a layer of drawables with the same renderer (basically, it's a Canvas element, SVG
 * element, or some type of DOM container). Doesn't strictly have to have its DOM element used directly (Canvas block
 * used for caches).  This type is abstract, and meant to be subclassed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import cleanArray from '../../../phet-core/js/cleanArray.js';
import { Drawable, scenery } from '../imports.js';

class Block extends Drawable {
  /**
   * @public
   *
   * @param {Display} display
   * @param {number} renderer
   */
  initialize( display, renderer ) {
    assert && assert( !display._isDisposing, 'Should not create a block for a Display that is being disposed of' );
    assert && assert( !display._isDisposed, 'Should not create a block for a disposed Display' );

    super.initialize( renderer );

    // @public {Display}
    this.display = display;

    // @public {number}
    this.drawableCount = 0;

    // @public {boolean} - flag handled in the stitch
    this.used = true;

    // @public {Drawable|null}
    this.firstDrawable = null;
    this.lastDrawable = null;
    this.pendingFirstDrawable = null;
    this.pendingLastDrawable = null;

    // @public {Block|null} - linked-list handling for blocks
    this.previousBlock = null;
    this.nextBlock = null;

    // @public {number} - last set z-index, valid if > 0.
    this.zIndex = 0;

    if ( assertSlow ) {
      this.drawableList = cleanArray( this.drawableList );
    }
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    assert && assert( this.drawableCount === 0, 'There should be no drawables on a block when it is disposed' );

    // clear references
    this.display = null;
    this.firstDrawable = null;
    this.lastDrawable = null;
    this.pendingFirstDrawable = null;
    this.pendingLastDrawable = null;

    this.previousBlock = null;
    this.nextBlock = null;

    // TODO: are we potentially leaking drawable lists here?
    if ( assertSlow ) {
      cleanArray( this.drawableList );
    }

    super.dispose();
  }

  /**
   * Adds a drawable to this block.
   * @public
   *
   * @param {Drawable} drawable
   */
  addDrawable( drawable ) {
    this.drawableCount++;
    this.markDirtyDrawable( drawable );

    if ( assertSlow ) {
      const idx = _.indexOf( this.drawableList, drawable );
      assertSlow && assertSlow( idx === -1, 'Drawable should not be added when it has not been removed' );
      this.drawableList.push( drawable );

      assertSlow && assertSlow( this.drawableCount === this.drawableList.length, 'Count sanity check, to make sure our assertions are not buggy' );
    }
  }

  /**
   * Removes a drawable from this block.
   * @public
   *
   * @param {Drawable} drawable
   */
  removeDrawable( drawable ) {
    this.drawableCount--;
    this.markDirty();

    if ( assertSlow ) {
      const idx = _.indexOf( this.drawableList, drawable );
      assertSlow && assertSlow( idx !== -1, 'Drawable should be already added when it is removed' );
      this.drawableList.splice( idx, 1 );

      assertSlow && assertSlow( this.drawableCount === this.drawableList.length, 'Count sanity check, to make sure our assertions are not buggy' );
    }
  }

  /**
   * @protected
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  onIntervalChange( firstDrawable, lastDrawable ) {
    // stub, should be filled in with behavior in blocks
  }

  /**
   * @public
   */
  updateInterval() {
    if ( this.pendingFirstDrawable !== this.firstDrawable ||
         this.pendingLastDrawable !== this.lastDrawable ) {
      this.onIntervalChange( this.pendingFirstDrawable, this.pendingLastDrawable );

      this.firstDrawable = this.pendingFirstDrawable;
      this.lastDrawable = this.pendingLastDrawable;
    }
  }

  /**
   * @public
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   */
  notifyInterval( firstDrawable, lastDrawable ) {
    this.pendingFirstDrawable = firstDrawable;
    this.pendingLastDrawable = lastDrawable;

    this.updateInterval();
  }

  /**
   * Runs checks on the drawable, based on certain flags.
   * @public
   * @override
   *
   * @param {boolean} allowPendingBlock
   * @param {boolean} allowPendingList
   * @param {boolean} allowDirty
   */
  audit( allowPendingBlock, allowPendingList, allowDirty ) {
    if ( assertSlow ) {
      super.audit( allowPendingBlock, allowPendingList, allowDirty );

      let count = 0;

      if ( !allowPendingList ) {

        // audit children, and get a count
        for ( let drawable = this.firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {
          drawable.audit( allowPendingBlock, allowPendingList, allowDirty );
          count++;
          if ( drawable === this.lastDrawable ) { break; }
        }

        if ( !allowPendingBlock ) {
          assertSlow && assertSlow( count === this.drawableCount, 'drawableCount should match' );

          assertSlow && assertSlow( this.firstDrawable === this.pendingFirstDrawable, 'No pending first drawable' );
          assertSlow && assertSlow( this.lastDrawable === this.pendingLastDrawable, 'No pending last drawable' );

          // scan through to make sure our drawable lists are identical
          for ( let d = this.firstDrawable; d !== null; d = d.nextDrawable ) {
            assertSlow && assertSlow( d.renderer === this.renderer, 'Renderers should match' );
            assertSlow && assertSlow( d.parentDrawable === this, 'This block should be this drawable\'s parent' );
            assertSlow && assertSlow( _.indexOf( this.drawableList, d ) >= 0 );
            if ( d === this.lastDrawable ) { break; }
          }
        }
      }
    }
  }
}

scenery.register( 'Block', Block );
export default Block;