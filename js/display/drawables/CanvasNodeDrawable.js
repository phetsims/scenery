// Copyright 2016-2022, University of Colorado Boulder

/**
 * Canvas drawable for CanvasNode. A generated CanvasSelfDrawable whose purpose will be drawing our CanvasNode.
 * One of these drawables will be created for each displayed instance of a CanvasNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasSelfDrawable, scenery } from '../../imports.js';

const emptyArray = []; // constant, used for line-dash

class CanvasNodeDrawable extends CanvasSelfDrawable {
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
    assert && assert( !node.selfBounds.isEmpty(), `${'CanvasNode should not be used with an empty canvasBounds. ' +
                                                     'Please set canvasBounds (or use setCanvasBounds()) on '}${node.constructor.name}` );

    if ( !node.selfBounds.isEmpty() ) {
      const context = wrapper.context;
      context.save();

      // set back to Canvas default styles
      // TODO: are these necessary, or can we drop them for performance?
      context.fillStyle = 'black';
      context.strokeStyle = 'black';
      context.lineWidth = 1;
      context.lineCap = 'butt';
      context.lineJoin = 'miter';
      context.lineDash = emptyArray;
      context.lineDashOffset = 0;
      context.miterLimit = 10;

      node.paintCanvas( context );

      context.restore();
    }
  }
}

scenery.register( 'CanvasNodeDrawable', CanvasNodeDrawable );

Poolable.mixInto( CanvasNodeDrawable );

export default CanvasNodeDrawable;