// Copyright 2015-2020, University of Colorado Boulder

/**
 * Shows the bounds of current fitted blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Shape from '../../../kite/js/Shape.js';
import inherit from '../../../phet-core/js/inherit.js';
import scenery from '../scenery.js';
import ShapeBasedOverlay from './ShapeBasedOverlay.js';

function FittedBlockBoundsOverlay( display, rootNode ) {
  ShapeBasedOverlay.call( this, display, rootNode, 'canvasNodeBoundsOverlay' );
}

scenery.register( 'FittedBlockBoundsOverlay', FittedBlockBoundsOverlay );

inherit( ShapeBasedOverlay, FittedBlockBoundsOverlay, {
  // @override
  addShapes: function() {
    const self = this;

    function processBackbone( backbone, matrix ) {
      if ( backbone.willApplyTransform ) {
        matrix = matrix.timesMatrix( backbone.backboneInstance.relativeTransform.matrix );
      }
      backbone.blocks.forEach( function( block ) {
        processBlock( block, matrix );
      } );
    }

    function processBlock( block, matrix ) {
      if ( block.fitBounds && !block.fitBounds.isEmpty() ) {
        self.addShape( Shape.bounds( block.fitBounds ).transformed( matrix ), 'rgba(255,0,0,0.8)', true );
      }
      if ( block.firstDrawable && block.lastDrawable ) {
        for ( let childDrawable = block.firstDrawable; childDrawable !== block.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          processDrawable( childDrawable, matrix );
        }
        processDrawable( block.lastDrawable, matrix );
      }
    }

    function processDrawable( drawable, matrix ) {
      // How we detect backbones (for now)
      if ( drawable.backboneInstance ) {
        processBackbone( drawable, matrix );
      }
    }

    processBackbone( this.display.rootBackbone, Matrix3.IDENTITY );
  }
} );

export default FittedBlockBoundsOverlay;