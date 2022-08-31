// Copyright 2019-2022, University of Colorado Boulder

/**
 * Canvas drawable for Sprites nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasSelfDrawable, Imageable, Node, scenery, SpriteInstanceTransformType } from '../../imports.js';

class SpritesCanvasDrawable extends CanvasSelfDrawable {
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
  paintCanvas( wrapper, node, matrix ) {
    assert && assert( node instanceof Node );

    const baseMipmapScale = Imageable.getApproximateMatrixScale( matrix ) * ( window.devicePixelRatio || 1 );

    const numInstances = node._spriteInstances.length;
    for ( let i = 0; i < numInstances; i++ ) {
      const spriteInstance = node._spriteInstances[ i ];
      const spriteImage = spriteInstance.sprite.imageProperty.value;
      const hasOpacity = spriteInstance.alpha !== 1 || spriteImage.imageOpacity !== 1;
      const hasMipmaps = spriteImage._mipmap && spriteImage.hasMipmaps();

      // If it includes opacity, we'll do a context save/restore
      if ( hasOpacity ) {
        wrapper.context.save();
        wrapper.context.globalAlpha *= spriteInstance.alpha * spriteImage.imageOpacity;
      }

      // If it's a translation only, we can add the offsets to the drawImage call directly (higher performance)
      if ( spriteInstance.transformType === SpriteInstanceTransformType.TRANSLATION && matrix.isTranslation() ) {
        if ( hasMipmaps ) {
          const level = spriteImage.getMipmapLevelFromScale( baseMipmapScale, Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT );
          const canvas = spriteImage.getMipmapCanvas( level );
          const multiplier = Math.pow( 2, level );
          wrapper.context.drawImage(
            canvas,
            spriteInstance.matrix.m02() - spriteImage.offset.x,
            spriteInstance.matrix.m12() - spriteImage.offset.y,
            canvas.width * multiplier,
            canvas.height * multiplier
          );
        }
        else {
          wrapper.context.drawImage(
            spriteImage.image,
            spriteInstance.matrix.m02() - spriteImage.offset.x,
            spriteInstance.matrix.m12() - spriteImage.offset.y
          );
        }
      }
      else {
        wrapper.context.save();
        spriteInstance.matrix.canvasAppendTransform( wrapper.context );

        if ( hasMipmaps ) {
          const level = spriteImage.getMipmapLevelFromScale( baseMipmapScale * Imageable.getApproximateMatrixScale( spriteInstance.matrix ), Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT );
          const canvas = spriteImage.getMipmapCanvas( level );
          const multiplier = Math.pow( 2, level );
          wrapper.context.drawImage(
            canvas,
            -spriteImage.offset.x,
            -spriteImage.offset.y,
            canvas.width * multiplier,
            canvas.height * multiplier
          );
        }
        else {
          wrapper.context.drawImage(
            spriteImage.image,
            -spriteImage.offset.x,
            -spriteImage.offset.y
          );
        }

        wrapper.context.restore();
      }

      if ( hasOpacity ) {
        wrapper.context.restore();
      }
    }
  }
}

scenery.register( 'SpritesCanvasDrawable', SpritesCanvasDrawable );

Poolable.mixInto( SpritesCanvasDrawable );

export default SpritesCanvasDrawable;