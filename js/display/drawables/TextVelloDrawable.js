// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import { Shape } from '../../../../kite/js/imports.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasContextWrapper, PathStatefulDrawable, PhetEncoding, scenery, SourceImage, TextCanvasDrawable, VelloSelfDrawable } from '../../imports.js';
import ArialFont from '../vello/ArialFont.js';
import ArialBoldFont from '../vello/ArialBoldFont.js';

const fillEncodingCache = new Map();
// const strokeEncodingCache = new Map();

const flipMatrix = Matrix3.rowMajor(
  1, 0, 0,
  0, -1, 0, // vertical flip
  0, 0, 1
);

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

    // TODO: use glyph detection to see if we have everything for the text!! (will need to determine font first)
    const useSwash = window.phet?.chipper?.queryParameters?.swashText;

    if ( !useSwash ) {
      // TODO: pooling, but figure out if we need to wait for the device.queue.onSubmittedWorkDone()
      const canvas = document.createElement( 'canvas' );

      // NOTE: getting value directly, so we don't set off any bounds validation during rendering
      // TODO: is 5px enough? too much?
      const selfBounds = node.selfBoundsProperty._value;
      if ( selfBounds.isValid() && selfBounds.hasNonzeroArea() ) {
        const bounds = node.selfBoundsProperty._value.transformed( matrix ).dilate( 5 ).roundOut();
        canvas.width = bounds.width * window.devicePixelRatio;
        canvas.height = bounds.height * window.devicePixelRatio;

        // TODO: clip this to the block's Canvas, so HUGE text won't create a huge texture
        const context = canvas.getContext( '2d' );
        context.scale( window.devicePixelRatio, window.devicePixelRatio );
        context.translate( -bounds.minX, -bounds.minY );
        matrix.canvasAppendTransform( context );

        const wrapper = new CanvasContextWrapper( canvas, context );

        // TODO: This is used in multiple places. Make it an actual static method, so we don't keep hacking it
        // TODO: also, this isn't the right matrix???
        TextCanvasDrawable.prototype.paintCanvas( wrapper, node, matrix );

        // TODO: faster function, don't create an Affine
        this.encoding.encode_matrix( Matrix3.rowMajor(
          1 / window.devicePixelRatio, 0, bounds.minX,
          0, 1 / window.devicePixelRatio, bounds.minY,
          0, 0, 1
        ) );
        // this.encoding.encode_matrix( Matrix3.translation( bounds.minX, bounds.minY ) );
        this.encoding.encode_linewidth( -1 );

        // TODO: faster "rect"
        const shape = Shape.rect( 0, 0, canvas.width, canvas.height );
        this.encoding.encode_kite_shape( shape, true, true, 100 );

        this.encoding.encode_image( new SourceImage( canvas.width, canvas.height, canvas ) );
      }
    }
    else {
      if ( node.hasFill() || node.hasStroke() ) {
        const font = ( node._font.weight === 'bold' ? ArialBoldFont : ArialFont );

        const shapedText = font.shapeText( node.renderedText, true );

        // TODO: stroking also!!!
        if ( node.hasFill() ) {

          let hasEncodedGlyph = false;

          // TODO: more performance possible easily
          const scale = node._font.numericSize / font.unitsPerEM;
          const sizedMatrix = matrix.timesMatrix( Matrix3.scaling( scale ) );

          // TODO: support this for text, so we can QUICKLY get the bounds of text
          // TODO: support these inside Font!!!

          let x = 0;
          shapedText.forEach( glyph => {
            const glyphMatrix = sizedMatrix.timesMatrix( Matrix3.translation( x + glyph.x, glyph.y ) ).timesMatrix( flipMatrix );
            x += glyph.advance;

            this.encoding.encode_matrix( glyphMatrix );
            this.encoding.encode_linewidth( -1 );

            let encoding = fillEncodingCache.get( glyph.shape );
            if ( !encoding ) {
              encoding = new PhetEncoding();
              encoding.encode_kite_shape( glyph.shape, true, false, 1 ); // TODO: tolerance
              fillEncodingCache.set( glyph.shape, encoding );
            }

            this.encoding.append( encoding );
            if ( !encoding.is_empty() ) {
              hasEncodedGlyph = true;
            }
          } );

          if ( hasEncodedGlyph ) {
            this.encoding.insert_path_marker();
            this.encoding.encode_paint( node.fill );
          }
        }
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