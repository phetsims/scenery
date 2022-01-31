// Copyright 2016-2021, University of Colorado Boulder

/**
 * Canvas drawable for Line nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { scenery, CanvasSelfDrawable, LineStatelessDrawable, Node } from '../../imports.js'; // eslint-disable-line

class LineCanvasDrawable extends LineStatelessDrawable( CanvasSelfDrawable ) {
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
    const context = wrapper.context;

    context.beginPath();
    context.moveTo( node._x1, node._y1 );
    context.lineTo( node._x2, node._y2 );

    if ( node.hasPaintableStroke() ) {
      node.beforeCanvasStroke( wrapper ); // defined in Paintable
      context.stroke();
      node.afterCanvasStroke( wrapper ); // defined in Paintable
    }
  }
}

scenery.register( 'LineCanvasDrawable', LineCanvasDrawable );

Poolable.mixInto( LineCanvasDrawable );

export default LineCanvasDrawable;