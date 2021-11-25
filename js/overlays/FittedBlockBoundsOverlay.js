// Copyright 2015-2020, University of Colorado Boulder

/**
 * Shows the bounds of current fitted blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Shape from '../../../kite/js/Shape.js';
import { scenery, ShapeBasedOverlay } from '../imports.js';

class FittedBlockBoundsOverlay extends ShapeBasedOverlay {
  /**
   * @param {Display} display
   * @param {Node} rootNode
   */
  constructor( display, rootNode ) {
    super( display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  /**
   * @public
   * @override
   */
  addShapes() {
    const self = this;

    function processBackbone( backbone, matrix ) {
      if ( backbone.willApplyTransform ) {
        matrix = matrix.timesMatrix( backbone.backboneInstance.relativeTransform.matrix );
      }
      backbone.blocks.forEach( block => {
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
}

scenery.register( 'FittedBlockBoundsOverlay', FittedBlockBoundsOverlay );
export default FittedBlockBoundsOverlay;