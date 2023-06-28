// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { imageBitmapMap, ImageStatefulDrawable, PhetEncoding, scenery, SourceImage, VelloSelfDrawable } from '../../imports.js';

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

    // TODO: implement imageOpacity

    this.encoding.reset( true );

    const node = this.node;
    const matrix = scalingMatrix.timesMatrix( this.instance.relativeTransform.matrix );

    // GPUImageCopyExternalImageSource
    let source;

    if ( node._image instanceof HTMLImageElement ) {
      const imageBitmap = imageBitmapMap.get( node._image );
      if ( imageBitmap && imageBitmap.width && imageBitmap.height ) {
        source = imageBitmap;
      }
      else {
        console.log( 'ImageVelloDrawable: image not loaded yet' );
      }
    }
    else if ( node._image instanceof HTMLCanvasElement ) {
      if ( node._image.width && node._image.height ) {
        source = node._image;
      }
    }

    // if we are not loaded yet, just ignore
    if ( source ) {
      this.encoding.encodeMatrix( matrix );
      this.encoding.encodeLineWidth( -1 );
      this.encoding.encodeRect( 0, 0, source.width, source.height );
      this.encoding.encodeImage( new SourceImage( source.width, source.height, source ) );
    }

    this.setToCleanState();

    return true;
  }
}

scenery.register( 'ImageVelloDrawable', ImageVelloDrawable );

Poolable.mixInto( ImageVelloDrawable );

export default ImageVelloDrawable;