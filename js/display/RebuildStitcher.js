// Copyright 2014-2022, University of Colorado Boulder

/**
 * Stitcher that rebuilds all of the blocks and reattaches drawables. Simple, but inefficient.
 *
 * Kept for now as a run-time comparison and baseline for the GreedyStitcher or any other more advanced (but
 * more error-prone) stitching process.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Renderer, scenery, Stitcher } from '../imports.js';

class RebuildStitcher extends Stitcher {
  /**
   * Main stitch entry point, called directly from the backbone or cache. We are modifying our backbone's blocks and
   * their attached drawables.
   * @public
   *
   * @param {BackboneDrawable} backbone
   * @param {Drawable|null} firstStitchDrawable
   * @param {Drawable|null} lastStitchDrawable
   * @param {Drawable|null} oldFirstStitchDrawable
   * @param {Drawable|null} oldLastStitchDrawable
   * @param {ChangeInterval} firstChangeInterval
   * @param {ChangeInterval} lastChangeInterval
   */
  stitch( backbone, firstDrawable, lastDrawable, oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval ) {
    this.initialize( backbone, firstDrawable, lastDrawable, oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval );

    for ( let d = backbone.previousFirstDrawable; d !== null; d = d.oldNextDrawable ) {
      this.notePendingRemoval( d );
      if ( d === backbone.previousLastDrawable ) { break; }
    }

    this.recordBackboneBoundaries();

    this.removeAllBlocks();

    let currentBlock = null;
    let currentRenderer = 0;
    let firstDrawableForBlock = null;

    // linked-list iteration inclusively from firstDrawable to lastDrawable
    for ( let drawable = firstDrawable; drawable !== null; drawable = drawable.nextDrawable ) {

      // if we need to switch to a new block, create it
      if ( !currentBlock || drawable.renderer !== currentRenderer ) {
        if ( currentBlock ) {
          this.notifyInterval( currentBlock, firstDrawableForBlock, drawable.previousDrawable );
        }

        currentRenderer = drawable.renderer;

        currentBlock = this.createBlock( currentRenderer, drawable );
        if ( Renderer.isDOM( currentRenderer ) ) {
          currentRenderer = 0;
        }

        this.appendBlock( currentBlock );

        firstDrawableForBlock = drawable;
      }

      this.notePendingAddition( drawable, currentBlock );

      // don't cause an infinite loop!
      if ( drawable === lastDrawable ) { break; }
    }
    if ( currentBlock ) {
      this.notifyInterval( currentBlock, firstDrawableForBlock, lastDrawable );
    }

    this.reindex();

    this.clean();
  }
}

scenery.register( 'RebuildStitcher', RebuildStitcher );

export default RebuildStitcher;