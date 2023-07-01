// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { Compose, FilterMatrix, imageBitmapMap, ImageStatefulDrawable, Mix, PhetEncoding, scenery, SourceImage, VelloSelfDrawable } from '../../imports.js';

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
    const matrix = scalingMatrix.timesMatrix( this.instance.relativeTransform.matrix );

    // GPUImageCopyExternalImageSource
    let source;

    if ( node._image instanceof HTMLImageElement ) {
      const imageBitmap = imageBitmapMap.get( node._image );
      if ( imageBitmap && imageBitmap.width && imageBitmap.height ) {
        source = imageBitmap;
      }
      else if ( node.imageWidth && node.imageHeight ) {
        // We generally end up here if the imageBitmap isn't computed yet (freshly drawn something!)
        // We'll try drawing it to a Canvas

        const canvas = document.createElement( 'canvas' );
        canvas.width = node.imageWidth;
        canvas.height = node.imageHeight;

        const context = canvas.getContext( '2d' );
        context.drawImage( node._image, 0, 0 );
        source = canvas;
      }
    }
    else if ( node._image instanceof HTMLCanvasElement ) {
      if ( node._image.width && node._image.height ) {
        source = node._image;
      }
    }

    const imageOpacity = node._imageOpacity;
    const needsImageOpacity = imageOpacity !== 1;

    // if we are not loaded yet, just ignore
    if ( source ) {
      // TODO: use an alpha parameter in the shaders once it is supported
      if ( needsImageOpacity ) {
        this.encoding.encodeMatrix( matrix );
        this.encoding.encodeLineWidth( -1 );
        this.encoding.encodeRect( 0, 0, source.width, source.height );

        this.encoding.encodeBeginClip( Mix.Normal, Compose.SrcOver, new FilterMatrix().multiplyAlpha( imageOpacity ) );
      }

      this.encoding.encodeMatrix( matrix );
      this.encoding.encodeLineWidth( -1 );
      this.encoding.encodeRect( 0, 0, source.width, source.height );
      this.encoding.encodeImage( new SourceImage( source.width, source.height, source ) );

      if ( needsImageOpacity ) {
        this.encoding.encodeEndClip();
      }
    }

    this.setToCleanState();

    return true;
  }
}

scenery.register( 'ImageVelloDrawable', ImageVelloDrawable );

Poolable.mixInto( ImageVelloDrawable );

export default ImageVelloDrawable;