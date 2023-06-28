// Copyright 2023, University of Colorado Boulder

/**
 * Vello drawable for Path nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { PathStatefulDrawable, PhetEncoding, scenery, VelloSelfDrawable } from '../../imports.js';

class PathVelloDrawable extends PathStatefulDrawable( VelloSelfDrawable ) {
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
  markDirtyRadius() { this.markDirtyShape(); }
  // @public
  markDirtyLine() { this.markDirtyShape(); }
  // @public
  markDirtyP1() { this.markDirtyShape(); }
  // @public
  markDirtyP2() { this.markDirtyShape(); }
  // @public
  markDirtyX1() { this.markDirtyShape(); }
  // @public
  markDirtyX2() { this.markDirtyShape(); }
  // @public
  markDirtyY1() { this.markDirtyShape(); }
  // @public
  markDirtyY2() { this.markDirtyShape(); }
  // @public
  markDirtyX() { this.markDirtyShape(); }
  // @public
  markDirtyY() { this.markDirtyShape(); }
  // @public
  markDirtyWidth() { this.markDirtyShape(); }
  // @public
  markDirtyHeight() { this.markDirtyShape(); }
  // @public
  markDirtyCornerXRadius() { this.markDirtyShape(); }
  // @public
  markDirtyCornerYRadius() { this.markDirtyShape(); }
  // @public
  markDirtyRectangle() { this.markDirtyShape(); }

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

    // TODO: consider caching the encoded shape, and only re-encoding if the shape changes. We should be able to
    // TODO: append out-of-order?

    this.encoding.reset( true );

    const node = this.node;
    const matrix = this.instance.relativeTransform.matrix;

    if ( node.shape ) {
      if ( node.hasFill() ) {
        this.encoding.encode_matrix( matrix );
        this.encoding.encode_linewidth( -1 );
        const numEncodedSegments = this.encoding.encode_kite_shape( node.shape, true, true, 1 );
        if ( numEncodedSegments ) {
          this.encoding.encode_paint( node.fill );
        }
      }
      if ( node.hasStroke() ) {
        this.encoding.encode_matrix( matrix );
        let shape = node.shape;
        if ( node.lineDash.length ) {
          // TODO: cache dashed shapes? OR AT LEAST DO NOT UPDATE THE IMAGE ENCODNG IF IT IS THE SAME
          // TODO: See SVG example
          shape = node.shape.getDashedShape( node.lineDash, node.lineDashOffset );
        }
        this.encoding.encode_linewidth( node.lineWidth );
        const numEncodedSegments = this.encoding.encode_kite_shape( shape, false, true, 1 );
        if ( numEncodedSegments ) {
          this.encoding.encode_paint( node.stroke );
        }
      }
    }

    this.setToCleanState();
    this.cleanPaintableState();

    return true;
  }
}

scenery.register( 'PathVelloDrawable', PathVelloDrawable );

Poolable.mixInto( PathVelloDrawable );

export default PathVelloDrawable;