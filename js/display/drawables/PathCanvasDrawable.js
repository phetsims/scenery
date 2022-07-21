// Copyright 2016-2022, University of Colorado Boulder

/**
 * Canvas drawable for Path nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasSelfDrawable, PaintableStatelessDrawable, scenery } from '../../imports.js';

class PathCanvasDrawable extends PaintableStatelessDrawable( CanvasSelfDrawable ) {
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

    if ( node.hasShape() ) {
      // TODO: fill/stroke delay optimizations?
      context.beginPath();
      node._shape.writeToContext( context );

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
  }

  /**
   * @public
   */
  markDirtyShape() {
    this.markPaintDirty();
  }
}

scenery.register( 'PathCanvasDrawable', PathCanvasDrawable );

Poolable.mixInto( PathCanvasDrawable );

export default PathCanvasDrawable;