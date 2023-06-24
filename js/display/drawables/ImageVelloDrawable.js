// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Shape } from '../../../../kite/js/imports.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { ImageStatefulDrawable, scenery, VelloSelfDrawable } from '../../imports.js';
import { Affine } from '../vello/Affine.js';
import { BufferImage } from '../vello/BufferImage.js';
import PhetEncoding from '../vello/PhetEncoding.js';

class ImageVelloDrawable extends ImageStatefulDrawable( VelloSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    this.encoding = new PhetEncoding();

    this.transformDirty = true;
  }

  /**
   * @public
   * @override
   */
  markTransformDirty() {
    this.transformDirty = true;

    super.markTransformDirty();
  }

  /**
   * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
   * @public
   * @override
   *
   * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
   *                      be done).
   */
  update() {
    // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
    if ( !super.update() ) {
      return false;
    }

    // TODO: only re-encode the image when IT changes, not when the transform changes

    this.encoding.reset( true );

    const matrixToAffine = matrix => new Affine( matrix.m00(), matrix.m10(), matrix.m01(), matrix.m11(), matrix.m02(), matrix.m12() );

    const node = this.node;
    const matrix = this.instance.relativeTransform.matrix;

    const canvas = document.createElement( 'canvas' );
    canvas.width = node.getImageWidth();
    canvas.height = node.getImageHeight();

    // if we are not loaded yet, just ignore
    if ( canvas.width && canvas.height ) {
      const context = canvas.getContext( '2d' );
      context.drawImage( node._image, 0, 0 );

      const imageData = context.getImageData( 0, 0, canvas.width, canvas.height );
      const buffer = new Uint8Array( imageData.data.buffer ).buffer; // copy in case the length isn't correct

      this.encoding.encode_transform( matrixToAffine( matrix ) );
      this.encoding.encode_linewidth( -1 );

      const shape = Shape.rect( 0, 0, canvas.width, canvas.height );

      this.encoding.encode_kite_shape( shape, true, true, 100 );

      this.encoding.encode_image( new BufferImage( canvas.width, canvas.height, buffer ) );
    }

    this.setToCleanState();

    return true;
  }
}

scenery.register( 'ImageVelloDrawable', ImageVelloDrawable );

Poolable.mixInto( ImageVelloDrawable );

export default ImageVelloDrawable;