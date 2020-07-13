// Copyright 2019-2020, University of Colorado Boulder

/**
 * Canvas drawable for Sprites nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import inherit from '../../../../phet-core/js/inherit.js';
import scenery from '../../scenery.js';
import SpriteInstance from '../../util/SpriteInstance.js';
import CanvasSelfDrawable from '../CanvasSelfDrawable.js';

/**
 * A generated CanvasSelfDrawable whose purpose will be drawing our sprites.
 * @constructor
 *
 * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
 * @param {Instance} instance
 */
function SpritesCanvasDrawable( renderer, instance ) {
  this.initializeCanvasSelfDrawable( renderer, instance );
}

scenery.register( 'SpritesCanvasDrawable', SpritesCanvasDrawable );

inherit( CanvasSelfDrawable, SpritesCanvasDrawable, {
  /**
   * Paints this drawable to a Canvas (the wrapper contains both a Canvas reference and its drawing context).
   * @public
   *
   * Assumes that the Canvas's context is already in the proper local coordinate frame for the node, and that any
   * other required effects (opacity, clipping, etc.) have already been prepared.
   *
   * This is part of the CanvasSelfDrawable API required to be implemented for subtypes.
   *
   * @param {CanvasContextWrapper} wrapper - Contains the Canvas and its drawing context
   * @param {Node} node - Our node that is being drawn
   * @param {Matrix3} matrix - The transformation matrix applied for this node's coordinate system.
   */
  paintCanvas: function( wrapper, node, matrix ) {

    const numInstances = node._spriteInstances.length;
    for ( let i = 0; i < numInstances; i++ ) {
      const spriteInstance = node._spriteInstances[ i ];
      const spriteImage = spriteInstance.sprite.imageProperty.value;
      const hasOpacity = spriteInstance.alpha !== 1;

      // If it includes opacity, we'll do a context save/restore
      if ( hasOpacity ) {
        wrapper.context.save();
        wrapper.context.globalAlpha *= spriteInstance.alpha;
      }

      // If it's a translation only, we can add the offsets to the drawImage call directly (higher performance)
      if ( spriteInstance.translationType === SpriteInstance.TRANSLATION ) {
        wrapper.context.drawImage(
          spriteImage.image,
          spriteInstance.matrix.m02() - spriteImage.offset.x,
          spriteInstance.matrix.m12() - spriteImage.offset.y
        );
      }
      else {
        wrapper.context.save();
        spriteInstance.matrix.canvasAppendTransform( wrapper.context );

        wrapper.context.drawImage(
          spriteImage.image,
          -spriteImage.offset.x,
          -spriteImage.offset.y
        );

        wrapper.context.restore();
      }

      if ( hasOpacity ) {
        wrapper.context.restore();
      }
    }
  }
} );

Poolable.mixInto( SpritesCanvasDrawable );

export default SpritesCanvasDrawable;