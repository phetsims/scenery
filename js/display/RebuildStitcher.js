// Copyright 2014-2019, University of Colorado Boulder

/**
 * Stitcher that rebuilds all of the blocks and reattaches drawables. Simple, but inefficient.
 *
 * Kept for now as a run-time comparison and baseline for the GreedyStitcher or any other more advanced (but
 * more error-prone) stitching process.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const inherit = require( 'PHET_CORE/inherit' );
  const Renderer = require( 'SCENERY/display/Renderer' );
  const scenery = require( 'SCENERY/scenery' );
  const Stitcher = require( 'SCENERY/display/Stitcher' );

  const prototype = {
    stitch: function( backbone, firstDrawable, lastDrawable, oldFirstDrawable, oldLastDrawable, firstChangeInterval, lastChangeInterval ) {
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
  };

  const RebuildStitcher = inherit( Stitcher, function RebuildStitcher() {
    // nothing done
  }, prototype );
  scenery.register( 'RebuildStitcher', RebuildStitcher );

  RebuildStitcher.stitchPrototype = prototype;

  return RebuildStitcher;
} );
