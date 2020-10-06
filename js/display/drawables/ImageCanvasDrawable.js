// Copyright 2016-2020, University of Colorado Boulder

/**
 * Canvas drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import inherit from '../../../../phet-core/js/inherit.js';
import Imageable from '../../nodes/Imageable.js';
import scenery from '../../scenery.js';
import CanvasSelfDrawable from '../CanvasSelfDrawable.js';

/**
 * A generated CanvasSelfDrawable whose purpose will be drawing our Image. One of these drawables will be created
 * for each displayed instance of a Image.
 * @constructor
 *
 * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
 * @param {Instance} instance
 */
function ImageCanvasDrawable( renderer, instance ) {
  this.initializeCanvasSelfDrawable( renderer, instance );
}

scenery.register( 'ImageCanvasDrawable', ImageCanvasDrawable );

inherit( CanvasSelfDrawable, ImageCanvasDrawable, {
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
        wrapper.context.drawImage( node._image, 0, 0 );
      }

      if ( hasImageOpacity ) {
        wrapper.context.restore();
      }
    }
  },

  // stateless dirty functions
  markDirtyImage: function() { this.markPaintDirty(); },
  markDirtyMipmap: function() { this.markPaintDirty(); },
  markDirtyImageOpacity: function() { this.markPaintDirty(); }
} );

Poolable.mixInto( ImageCanvasDrawable );

export default ImageCanvasDrawable;