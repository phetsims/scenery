// Copyright 2016-2026, University of Colorado Boulder

/**
 * Canvas drawable for Image nodes.
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import CanvasSelfDrawable from '../../display/CanvasSelfDrawable.js';
import Imageable from '../../nodes/Imageable.js';
import scenery from '../../scenery.js';

class ImageCanvasDrawable extends CanvasSelfDrawable {
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
   * @param {scenery.Node} node - Our node that is being drawn
   * @param {Matrix3} matrix - The transformation matrix applied for this node's coordinate system.
   */
  paintCanvas( wrapper, node, matrix ) {
    const hasImageOpacity = node._imageOpacity !== 1;

    // Ensure that the image has been loaded by checking whether it has a width or height of 0.
    // See https://github.com/phetsims/scenery/issues/536
    if ( node._image && node._image.width !== 0 && node._image.height !== 0 ) {
      // If we have image opacity, we need to apply the opacity on top of whatever globalAlpha may exist
      if ( hasImageOpacity ) {
        wrapper.context.save();
        wrapper.context.globalAlpha *= node._imageOpacity;
      }

      if ( node._mipmap && node.hasMipmaps() ) {
        const level = node.getMipmapLevel( matrix, Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT );
        const canvas = node.getMipmapCanvas( level );
        const multiplier = Math.pow( 2, level );
        wrapper.context.drawImage( canvas, 0, 0, canvas.width * multiplier, canvas.height * multiplier );
      }
      else {

        // Use an explicit destination size so Canvas rendering matches the SVG/DOM drawables, which set explicit
        // width/height attributes from getImageWidth()/getImageHeight(). For raster images this matches the intrinsic
        // size, so behavior is the same as in the 3-arg call. For SVG images with a viewBox but no width/height
        // attributes, the intrinsic size is undefined per the CSS spec and browser-dependent, so the 3-argument
        // drawImage form could draw the image at a different scale than the Node's bounds (e.g. in screenshots, which
        // render through this code path).
        // See https://github.com/phetsims/quantum-wave-interference/issues/226
        wrapper.context.drawImage( node._image, 0, 0, node.getImageWidth(), node.getImageHeight() );
      }

      if ( hasImageOpacity ) {
        wrapper.context.restore();
      }
    }
  }

  /**
   * @public
   */
  markDirtyImage() {
    this.markPaintDirty();
  }

  /**
   * @public
   */
  markDirtyMipmap() {
    this.markPaintDirty();
  }

  /**
   * @public
   */
  markDirtyImageOpacity() {
    this.markPaintDirty();
  }
}

scenery.register( 'ImageCanvasDrawable', ImageCanvasDrawable );

Poolable.mixInto( ImageCanvasDrawable );

export default ImageCanvasDrawable;