// Copyright 2015-2022, University of Colorado Boulder

/**
 * Shows the bounds of current fitted blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import { Shape } from '../../../kite/js/imports.js';
import { scenery, ShapeBasedOverlay, Display, Node, BackboneDrawable, Block, FittedBlock, Drawable, IOverlay } from '../imports.js';

class FittedBlockBoundsOverlay extends ShapeBasedOverlay implements IOverlay {
  constructor( display: Display, rootNode: Node ) {
    super( display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  addShapes() {
    const self = this;

    function processBackbone( backbone: BackboneDrawable, matrix: Matrix3 ) {
      if ( backbone.willApplyTransform ) {
        matrix = matrix.timesMatrix( backbone.backboneInstance.relativeTransform.matrix );
      }
      backbone.blocks.forEach( ( block: Block ) => {
        processBlock( block, matrix );
      } );
    }

    function processBlock( block: Block, matrix: Matrix3 ) {
      if ( block instanceof FittedBlock && !block.fitBounds!.isEmpty() ) {
        self.addShape( Shape.bounds( block.fitBounds! ).transformed( matrix ), 'rgba(255,0,0,0.8)', true );
      }
      if ( block.firstDrawable && block.lastDrawable ) {
        for ( let childDrawable = block.firstDrawable; childDrawable !== block.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          processDrawable( childDrawable, matrix );
        }
        processDrawable( block.lastDrawable, matrix );
      }
    }

    function processDrawable( drawable: Drawable, matrix: Matrix3 ) {
      // How we detect backbones (for now)
      if ( drawable instanceof BackboneDrawable ) {
        processBackbone( drawable, matrix );
      }
    }

    processBackbone( this.display.rootBackbone, Matrix3.IDENTITY );
  }
}

scenery.register( 'FittedBlockBoundsOverlay', FittedBlockBoundsOverlay );
export default FittedBlockBoundsOverlay;