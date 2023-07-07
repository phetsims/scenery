// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { Imageable, ImageStatefulDrawable, PhetEncoding, scenery, SourceImage, VelloSelfDrawable } from '../../imports.js';

const scalingMatrix = Matrix3.scaling( window.devicePixelRatio );

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

    // TODO: only re-encode the image when IT changes, not when the transform changes!!!
    // TODO: This is fairly important for performance it seems

    this.encoding.reset( true );

    const node = this.node;
    let matrix = scalingMatrix.timesMatrix( this.instance.relativeTransform.matrix );

    let source;
    if ( node._mipmap && node.hasMipmaps() ) {
      const level = node.getMipmapLevel( matrix, Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT );
      source = node.getMipmapCanvas( level );
      matrix = matrix.timesMatrix( Matrix3.scaling( Math.pow( 2, level ) ) );
    }
    else {
      source = PhetEncoding.getSourceFromImage( node._image, node.imageWidth, node.imageHeight );
    }

    // if we are not loaded yet, just ignore
    if ( source ) {
      this.encoding.encodeMatrix( matrix );
      this.encoding.encodeLineWidth( -1 );
      this.encoding.encodeRect( 0, 0, source.width, source.height );
      this.encoding.encodeImage( new SourceImage( source.width, source.height, source ), node._imageOpacity );
    }

    this.setToCleanState();

    return true;
  }
}

scenery.register( 'ImageVelloDrawable', ImageVelloDrawable );

Poolable.mixInto( ImageVelloDrawable );

export default ImageVelloDrawable;