// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasContextWrapper, PathStatefulDrawable, PhetEncoding, scenery, SourceImage, Utils, TextCanvasDrawable, VelloSelfDrawable } from '../../imports.js';
import ArialBoldFont from '../vello/ArialBoldFont.js';
import ArialFont from '../vello/ArialFont.js';

const fillEncodingCache = new Map();
const strokeEncodingCache = new Map();

const flipMatrix = Matrix3.rowMajor(
  1, 0, 0,
  0, -1, 0, // vertical flip
  0, 0, 1
);

const scalingMatrix = Matrix3.scaling( window.devicePixelRatio );

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
    const matrix = scalingMatrix.timesMatrix( this.instance.relativeTransform.matrix );

    if ( node.hasFill() || node.hasStroke() ) {

      // TODO: font fallbacks!
      const font = ( node._font.weight === 'bold' ? ArialBoldFont : ArialFont );

      let useSwash = window.phet?.chipper?.queryParameters?.swashText;

      let shapedText;
      if ( useSwash ) {
        // Use a few "replacement" characters to see if we actually have some missing glyphs somehow
        // Otherwise it was rendering square boxes
        const badShapedText = font.shapeText( '\u25a1\ufffd', true );
        const badIDs = badShapedText.map( glyph => glyph.id );

        shapedText = font.shapeText( node.renderedText, true );

        // If we don't have all of the glyphs we'll need to render, fall back to the non-swash version
        // TODO: don't create closures like this
        if ( !shapedText || shapedText.some( glyph => badIDs.includes( glyph.id ) ) ) {
          useSwash = false;
        }
      }

      if ( !useSwash ) {
        // TODO: pooling, but figure out if we need to wait for the device.queue.onSubmittedWorkDone()
        const canvas = document.createElement( 'canvas' );
        canvas.style.position = 'absolute';
        canvas.style.left = '0';
        canvas.style.top = '0';
        canvas.style.pointerEvents = 'none';

        // Reset any fit transforms that were applied
        Utils.prepareForTransform( canvas ); // Apply CSS needed for future CSS transforms to work properly.
        Utils.unsetTransform( canvas ); // clear out any transforms that could have been previously applied

        // NOTE: getting value directly, so we don't set off any bounds validation during rendering
        // TODO: is 5px enough? too much?
        const selfBounds = node.selfBoundsProperty._value;
        if ( selfBounds.isValid() && selfBounds.hasNonzeroArea() ) {
          // TODO: only use accurate bounds?!!!
          const bounds = node.selfBoundsProperty._value.transformed( matrix ).dilate( 5 ).roundOut();
          canvas.width = bounds.width;
          canvas.height = bounds.height;
          canvas.style.width = `${bounds.width / window.devicePixelRatio}px`;
          canvas.style.height = `${bounds.height / window.devicePixelRatio}px`;

          // TODO: clip this to the block's Canvas, so HUGE text won't create a huge texture
          // TODO: Check... Ohm's Law?
          // TODO: NOTE: If a block resizes, WOULD we be marked as dirty? If not, we'd have to listen to it
          const context = canvas.getContext( '2d' );
          context.translate( -bounds.minX, -bounds.minY );
          matrix.canvasAppendTransform( context );

          TextCanvasDrawable.paintTextNodeToCanvas( new CanvasContextWrapper( canvas, context ), node, matrix );

          // TODO: faster function, don't create an object?
          this.encoding.encodeMatrix( Matrix3.translation( bounds.minX, bounds.minY ) );
          this.encoding.encodeLineWidth( -1 );
          this.encoding.encodeRect( 0, 0, canvas.width, canvas.height );
          this.encoding.encodeImage( new SourceImage( canvas.width, canvas.height, canvas ) );
        }
      }
      else {
        const scale = node._font.numericSize / font.unitsPerEM;
        const sizedMatrix = matrix.timesMatrix( Matrix3.scaling( scale ) );

        if ( node.hasFill() ) {
          this.encodeGlyphRun( shapedText, sizedMatrix, true, matrix );
        }
        if ( node.hasStroke() ) {
          this.encodeGlyphRun( shapedText, sizedMatrix, false, matrix );
        }
      }
    }


    this.setToCleanState();
    this.cleanPaintableState();

    return true;
  }

  // @private
  encodeGlyphRun( shapedText, sizedMatrix, isFill, matrix ) {
    let hasEncodedGlyph = false;

    const swashTextColor = window.phet?.chipper?.queryParameters?.swashTextColor;

    // TODO: support this for text, so we can QUICKLY get the bounds of text
    // TODO: support these inside Font!!!

    let x = 0;
    shapedText.forEach( glyph => {
      const glyphMatrix = sizedMatrix.timesMatrix( Matrix3.translation( x + glyph.x, glyph.y ) ).timesMatrix( flipMatrix );

      const encoding = TextVelloDrawable.getGlyphEncoding( glyph.shape, isFill );

      if ( !encoding.isEmpty() ) {
        this.encoding.encodeMatrix( glyphMatrix );
        this.encoding.encodeLineWidth( -1 );
        this.encoding.append( encoding );
        hasEncodedGlyph = true;
      }

      x += glyph.advance;
    } );

    if ( hasEncodedGlyph ) {
      this.encoding.insertPathMarker();
      this.encoding.encodePaint( swashTextColor ? swashTextColor : ( isFill ? this.node.fill : this.node.stroke ), matrix );
    }
  }

  // @private
  static getGlyphEncoding( shape, isFill ) {
    const cache = isFill ? fillEncodingCache : strokeEncodingCache;
    let encoding = cache.get( shape );
    if ( !encoding ) {
      encoding = new PhetEncoding();
      encoding.encodeShape( shape, isFill, false, 1 ); // TODO: tolerance
      cache.set( shape, encoding );
    }

    return encoding;
  }
}

scenery.register( 'TextVelloDrawable', TextVelloDrawable );

Poolable.mixInto( TextVelloDrawable );

export default TextVelloDrawable;