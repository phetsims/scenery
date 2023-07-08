// Copyright 2016-2022, University of Colorado Boulder

/**
 * Canvas drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Matrix3 from '../../../../dot/js/Matrix3.js';
import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasContextWrapper, CanvasSelfDrawable, PaintableStatelessDrawable, scenery, Utils } from '../../imports.js';

const canvas = document.createElement( 'canvas' );
canvas.style.position = 'absolute';
canvas.style.left = '0';
canvas.style.top = '0';
canvas.style.pointerEvents = 'none';

// @private {number} - unique ID so that we can support rasterization with Display.foreignObjectRasterization
canvas.id = 'scenery-canvasasrtarst';

// @private {CanvasRenderingContext2D}
const context = canvas.getContext( '2d' );
context.save(); // We always immediately save every Canvas so we can restore/save for clipping

// workaround for Chrome (WebKit) miterLimit bug: https://bugs.webkit.org/show_bug.cgi?id=108763
context.miterLimit = 20;
context.miterLimit = 10;
document.body.appendChild( canvas );

Utils.prepareForTransform( canvas ); // Apply CSS needed for future CSS transforms to work properly.
Utils.unsetTransform( canvas ); // clear out any transforms that could have been previously applied
class TextCanvasDrawable extends PaintableStatelessDrawable( CanvasSelfDrawable ) {
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
    TextCanvasDrawable.paintTextNodeToCanvas( wrapper, node );


    // NOTE: getting value directly, so we don't set off any bounds validation during rendering
    // TODO: is 5px enough? too much?
    const selfBounds = node.selfBoundsProperty._value;
    if ( selfBounds.isValid() && selfBounds.hasNonzeroArea() ) {
      // TODO: only use accurate bounds?!!!
      // const scalingMatrix = Matrix3.scaling( window.devicePixelRatio );
      // const matrix = scalingMatrix.timesMatrix( this.instance.relativeTransform.matrix );
      // const bounds = node.selfBoundsProperty._value.transformed( matrix ).dilate( 5 ).roundOut();
      canvas.width = wrapper.canvas.width;
      canvas.height = wrapper.canvas.height;
      // document.body.appendChild( canvas );
      // canvas.width = bounds.width;
      // canvas.height = bounds.height;

      // TODO: clip this to the block's Canvas, so HUGE text won't create a huge texture
      // TODO: Check... Ohm's Law?
      // TODO: NOTE: If a block resizes, WOULD we be marked as dirty? If not, we'd have to listen to it
      // const context = canvas.getContext( '2d' );
      // context.translate( -bounds.minX, -bounds.minY );
      // matrix.canvasAppendTransform( context );
      // context.setTransform(
      //   1.978124976158142,
      //   0,
      //   0,
      //   1.978124976158142,
      //   1303.6134033203125,
      //   1285.7547607421875
      // );

      TextCanvasDrawable.paintTextNodeToCanvas( new CanvasContextWrapper( canvas, context ), node, matrix );
    }

    if ( node.renderedText.includes( 'Earth Days' ) ) {
      if ( wrapper.canvas.toDataURL() !== canvas.toDataURL() ) {
        debugger;
      }
    }
  }

  /**
   * Paints this drawable to a Canvas (the wrapper contains both a Canvas reference and its drawing context).
   * @public
   *
   * Assumes that the Canvas's context is already in the proper local coordinate frame for the node, and that any
   * other required effects (opacity, clipping, etc.) have already been prepared.
   */
  static paintTextNodeToCanvas( wrapper, node ) {
    const context = wrapper.context;

    wrapper.resetStyles();

    context.reset();
    context.setTransform( 1, 0, 0, 1, 0, 0 ); // identity
    context.fillStyle = 'black';
    context.fillRect( 0, 0, wrapper.canvas.width, wrapper.canvas.height ); // fill with black (for debugging
    context.setTransform(
      1.978124976158142,
      0,
      0,
      1.978124976158142,
      1303.6134033203125,
      1285.7547607421875
    );

    // context.save(); // just in case we were clipping/etc.
    // context.clearRect( 0, 0, wrapper.canvas.width, wrapper.canvas.height ); // clear everything
    // context.restore();

    if ( context.textRendering ) {
      context.textRendering = 'geometricPrecision';
      // context.letterSpacing = '0px';
      // context.wordSpacing = '0px';
    }

    context.font = node._font.getFont();
    context.direction = 'ltr';

    // extra parameters we need to set, but should avoid setting if we aren't drawing anything
    // if ( node.hasFill() || node.hasPaintableStroke() ) {
    //   wrapper.setFont( node._font.getFont() );
    //   wrapper.setDirection( 'ltr' );
    // }

    context.fillStyle = 'white';
    context.fillText( node.renderedText, 0, 0 );

    // if ( node.hasFill() ) {
    //   // node.beforeCanvasFill( wrapper ); // defined in Paintable
    //   // node.afterCanvasFill( wrapper ); // defined in Paintable
    // }
    // if ( node.hasPaintableStroke() ) {
    //   node.beforeCanvasStroke( wrapper ); // defined in Paintable
    //   context.strokeText( node.renderedText, 0, 0 );
    //   node.afterCanvasStroke( wrapper ); // defined in Paintable
    // }

    if ( node.renderedText.includes( 'Earth Days' ) ) {
      // console.log( node.renderedText );
      // console.log( wrapper.context.getTransform() );
      console.log( wrapper.canvas.toDataURL() );
      // [ 'letterSpacing', 'wordSpacing', 'fillStyle', 'strokeStyle', 'filter', 'globalAlpha', 'globalCompositeOperation', 'lineWidth', 'lineCap', 'lineJoin', 'miterLimit', 'lineDashOffset', 'shadowOffsetX', 'shadowOffsetY', 'shadowBlur', 'shadowColor', 'font', 'textAlign', 'textBaseline', 'direction', 'fontKerning', 'fontStretch', 'fontVariantCaps', 'textRendering', 'imageSmoothingEnabled', 'imageSmoothingQuality' ].forEach( key => {
      //   console.log( `${key}: ${wrapper.context[ key ]}` );
      // } );
      console.log( wrapper.canvas.style );
    }
  }

  /**
   * @public
   */
  markDirtyText() {
    this.markPaintDirty();
  }

  /**
   * @public
   */
  markDirtyFont() {
    this.markPaintDirty();
  }

  /**
   * @public
   */
  markDirtyBounds() {
    this.markPaintDirty();
  }
}

scenery.register( 'TextCanvasDrawable', TextCanvasDrawable );

Poolable.mixInto( TextCanvasDrawable );

export default TextCanvasDrawable;