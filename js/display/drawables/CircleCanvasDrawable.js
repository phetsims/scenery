// Copyright 2016-2022, University of Colorado Boulder

/**
 * Canvas drawable for Circle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasSelfDrawable, Node, PaintableStatelessDrawable, scenery } from '../../imports.js';

class CircleCanvasDrawable extends PaintableStatelessDrawable( CanvasSelfDrawable ) {
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
   * @param {Node} node - Our node that is being drawn
   * @param {Matrix3} matrix - The transformation matrix applied for this node's coordinate system.
   */
  paintCanvas( wrapper, node, matrix ) {
    assert && assert( node instanceof Node );
    const context = wrapper.context;

    context.beginPath();
    context.arc( 0, 0, node._radius, 0, Math.PI * 2, false );
    context.closePath();

    if ( node.hasFill() ) {
      node.beforeCanvasFill( wrapper ); // defined in Paintable
      context.fill();
      node.afterCanvasFill( wrapper ); // defined in Paintable
    }
    if ( node.hasPaintableStroke() ) {
      node.beforeCanvasStroke( wrapper ); // defined in Paintable
      context.stroke();
      node.afterCanvasStroke( wrapper ); // defined in Paintable
    }
  }

  /**
   * Called when the radius of the circle changes.
   * @public
   */
  markDirtyRadius() {
    this.markPaintDirty();
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    super.dispose();
  }
}

scenery.register( 'CircleCanvasDrawable', CircleCanvasDrawable );

Poolable.mixInto( CircleCanvasDrawable );

export default CircleCanvasDrawable;