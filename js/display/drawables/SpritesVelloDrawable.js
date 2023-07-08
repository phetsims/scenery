// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Sprites nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { Imageable, PhetEncoding, scenery, SourceImage, VelloSelfDrawable } from '../../imports.js';

const scalingMatrix = Matrix3.scaling( window.devicePixelRatio );
const scratchMatrix1 = Matrix3.pool.fetch();
const scratchMatrix2 = Matrix3.pool.fetch();

class SpritesVelloDrawable extends VelloSelfDrawable {
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

    this.encoding.reset( true );

    const node = this.node;
    const baseMatrix = scalingMatrix.timesMatrix( this.instance.relativeTransform.matrix );

    const baseMipmapScale = Imageable.getApproximateMatrixScale( baseMatrix );

    const numInstances = node._spriteInstances.length;
    for ( let i = 0; i < numInstances; i++ ) {
      const spriteInstance = node._spriteInstances[ i ];
      const spriteImage = spriteInstance.sprite.imageProperty.value;
      const hasMipmaps = spriteImage._mipmap && spriteImage.hasMipmaps();

      const matrix = scratchMatrix1
        .set( baseMatrix )
        .multiplyMatrix( spriteInstance.matrix )
        .multiplyMatrix( scratchMatrix2.setToTranslation( -spriteImage.offset.x, -spriteImage.offset.y ) );

      let source;
      if ( hasMipmaps ) {
        const level = spriteImage.getMipmapLevelFromScale( baseMipmapScale * Imageable.getApproximateMatrixScale( spriteInstance.matrix ), Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT );
        source = spriteImage.getMipmapCanvas( level );
        matrix.multiplyMatrix( scratchMatrix2.setToScale( Math.pow( 2, level ) ) );
      }
      else {
        source = PhetEncoding.getSourceFromImage( spriteImage.image, spriteImage.image.width, spriteImage.image.height );
      }

      // if we are not loaded yet, just ignore
      if ( source ) {
        this.encoding.encodeMatrix( matrix );
        this.encoding.encodeLineWidth( -1 );
        this.encoding.encodeRect( 0, 0, source.width, source.height );
        this.encoding.encodeImage( new SourceImage( source.width, source.height, source ), spriteInstance.alpha );
      }
    }

    // let source;
    // if ( node._mipmap && node.hasMipmaps() ) {
    //   const level = node.getMipmapLevel( matrix, Imageable.CANVAS_MIPMAP_BIAS_ADJUSTMENT );
    //   source = node.getMipmapCanvas( level );
    //   matrix = matrix.timesMatrix( Matrix3.scaling( Math.pow( 2, level ) ) );
    // }
    // else {
    //   source = PhetEncoding.getSourceFromImage( node._image, node.imageWidth, node.imageHeight );
    // }

    return true;
  }
}

scenery.register( 'SpritesVelloDrawable', SpritesVelloDrawable );

Poolable.mixInto( SpritesVelloDrawable );

export default SpritesVelloDrawable;