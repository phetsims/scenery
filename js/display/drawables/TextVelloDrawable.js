// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

// @ts-expect-error yeah, this doesn't exist for most people
import { get_glyph, shape_text } from '../vello/swash.js';
import Matrix3 from '../../../../dot/js/Matrix3.js';
import { Shape } from '../../../../kite/js/imports.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { PathStatefulDrawable, scenery, VelloSelfDrawable } from '../../imports.js';
import { Affine } from '../vello/Affine.js';
import PhetEncoding from '../vello/PhetEncoding.js';

class TextVelloDrawable extends PathStatefulDrawable( VelloSelfDrawable ) {
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

  // Stubs since the nodes expect these to be defined
  // @public
  markDirtyText() { this.markDirtyShape(); }
  // @public
  markDirtyBounds() { this.markDirtyShape(); }
  // @public
  markDirtyFont() { this.markDirtyShape(); }

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
    const matrix = this.instance.relativeTransform.matrix;

    if ( node.hasFill() ) {
      const shapedText = JSON.parse( shape_text( node.renderedText, true ) );

      let hasEncodedGlyph = false;

      // TODO: more performance possible easily
      const scale = node._font.numericSize / 2048; // get UPM
      const sizedMatrix = matrix.timesMatrix( Matrix3.scaling( scale ) );
      const shearMatrix = Matrix3.rowMajor(
        // approx 14 degrees, with always vertical flip
        1, node._font.style !== 'normal' ? 0.2419 : 0, 0,
        0, -1, 0, // vertical flip
        0, 0, 1
      );

      let embolden = 0;
      if ( node._font.weight === 'bold' ) {
        embolden = 40;
      }

      let x = 0;
      shapedText.forEach( glyph => {
        const shape = new Shape( get_glyph( glyph.id, embolden, embolden ) ); // TODO: bold! (italic with oblique transform!!)

        // TODO: check whether the glyph y needs to be reversed! And italics/oblique
        const glyphMatrix = sizedMatrix.timesMatrix( Matrix3.translation( x + glyph.x, glyph.y ) ).timesMatrix( shearMatrix );
        x += glyph.adv;

        this.encoding.encode_transform( new Affine(
          glyphMatrix.m00(), glyphMatrix.m10(), glyphMatrix.m01(), glyphMatrix.m11(),
          glyphMatrix.m02(), glyphMatrix.m12()
        ) );
        this.encoding.encode_linewidth( -1 );
        const encodedCount = this.encoding.encode_kite_shape( shape, true, false, 1 );
        if ( encodedCount ) {
          hasEncodedGlyph = true;
        }
      } );

      if ( hasEncodedGlyph ) {
        this.encoding.insert_path_marker();
        this.encoding.encode_paint( node.fill );
      }
    }

    // TODO: more fine-grained dirtying (have a path encoding perhaps?
    // if ( this.paintDirty ) {
    //   //
    // }
    //
    // if ( this.dirtyShape ) {
    //   //
    // }

    this.setToCleanState();
    this.cleanPaintableState();

    return true;
  }
}

scenery.register( 'TextVelloDrawable', TextVelloDrawable );

Poolable.mixInto( TextVelloDrawable );

export default TextVelloDrawable;