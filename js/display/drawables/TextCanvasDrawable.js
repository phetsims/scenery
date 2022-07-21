// Copyright 2016-2022, University of Colorado Boulder

/**
 * Canvas drawable for Text nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasSelfDrawable, PaintableStatelessDrawable, scenery } from '../../imports.js';

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
    const context = wrapper.context;

    // extra parameters we need to set, but should avoid setting if we aren't drawing anything
    if ( node.hasFill() || node.hasPaintableStroke() ) {
      wrapper.setFont( node._font.getFont() );
      wrapper.setDirection( 'ltr' );
    }

    if ( node.hasFill() ) {
      node.beforeCanvasFill( wrapper ); // defined in Paintable
      context.fillText( node.renderedText, 0, 0 );
      node.afterCanvasFill( wrapper ); // defined in Paintable
    }
    if ( node.hasPaintableStroke() ) {
      node.beforeCanvasStroke( wrapper ); // defined in Paintable
      context.strokeText( node.renderedText, 0, 0 );
      node.afterCanvasStroke( wrapper ); // defined in Paintable
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