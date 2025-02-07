// Copyright 2015-2025, University of Colorado Boulder

/**
 * Shows the bounds of current fitted blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../dot/js/Matrix3.js';
import Shape from '../../../kite/js/Shape.js';
import BackboneDrawable from '../display/BackboneDrawable.js';
import Block from '../display/Block.js';
import type Display from '../display/Display.js';
import Drawable from '../display/Drawable.js';
import FittedBlock from '../display/FittedBlock.js';
import type Node from '../nodes/Node.js';
import ShapeBasedOverlay from '../overlays/ShapeBasedOverlay.js';
import type TOverlay from '../overlays/TOverlay.js';
import scenery from '../scenery.js';

export default class FittedBlockBoundsOverlay extends ShapeBasedOverlay implements TOverlay {
  public constructor( display: Display, rootNode: Node ) {
    super( display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  public addShapes(): void {

    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const selfReference = this; // eslint-disable-line consistent-this

    function processBackbone( backbone: BackboneDrawable, matrix: Matrix3 ): void {
      if ( backbone.willApplyTransform ) {
        matrix = matrix.timesMatrix( backbone.backboneInstance.relativeTransform.matrix );
      }
      backbone.blocks.forEach( ( block: Block ) => {
        processBlock( block, matrix );
      } );
    }

    function processBlock( block: Block, matrix: Matrix3 ): void {
      if ( block instanceof FittedBlock && !block.fitBounds!.isEmpty() ) {
        selfReference.addShape( Shape.bounds( block.fitBounds! ).transformed( matrix ), 'rgba(255,0,0,0.8)', true );
      }
      if ( block.firstDrawable && block.lastDrawable ) {
        for ( let childDrawable = block.firstDrawable; childDrawable !== block.lastDrawable; childDrawable = childDrawable.nextDrawable ) {
          processDrawable( childDrawable, matrix );
        }
        processDrawable( block.lastDrawable, matrix );
      }
    }

    function processDrawable( drawable: Drawable, matrix: Matrix3 ): void {
      // How we detect backbones (for now)
      if ( drawable instanceof BackboneDrawable ) {
        processBackbone( drawable, matrix );
      }
    }

    processBackbone( this.display.rootBackbone, Matrix3.IDENTITY );
  }
}

scenery.register( 'FittedBlockBoundsOverlay', FittedBlockBoundsOverlay );