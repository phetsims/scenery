// Copyright 2016-2022, University of Colorado Boulder

/**
 * Canvas drawable for Rectangle nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { CanvasSelfDrawable, PaintableStatelessDrawable, scenery } from '../../imports.js';

class RectangleCanvasDrawable extends PaintableStatelessDrawable( CanvasSelfDrawable ) {
  /**
   * Convenience function for drawing a rectangular path (with our Rectangle node's parameters) to the Canvas context.
   * @private
   *
   * @param {CanvasRenderingContext2D} context - To execute drawing commands on.
   * @param {Node} node - The node whose rectangle we want to draw
   */
  writeRectangularPath( context, node ) {
    context.beginPath();
    context.moveTo( node._rectX, node._rectY );
    context.lineTo( node._rectX + node._rectWidth, node._rectY );
    context.lineTo( node._rectX + node._rectWidth, node._rectY + node._rectHeight );
    context.lineTo( node._rectX, node._rectY + node._rectHeight );
    context.closePath();
  }

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

    // use the standard version if it's a rounded rectangle, since there is no Canvas-optimized version for that
    if ( node.isRounded() ) {
      context.beginPath();
      const maximumArcSize = node.getMaximumArcSize();
      const arcw = Math.min( node._cornerXRadius, maximumArcSize );
      const arch = Math.min( node._cornerYRadius, maximumArcSize );
      const lowX = node._rectX + arcw;
      const highX = node._rectX + node._rectWidth - arcw;
      const lowY = node._rectY + arch;
      const highY = node._rectY + node._rectHeight - arch;
      if ( arcw === arch ) {
        // we can use circular arcs, which have well defined stroked offsets
        context.arc( highX, lowY, arcw, -Math.PI / 2, 0, false );
        context.arc( highX, highY, arcw, 0, Math.PI / 2, false );
        context.arc( lowX, highY, arcw, Math.PI / 2, Math.PI, false );
        context.arc( lowX, lowY, arcw, Math.PI, Math.PI * 3 / 2, false );
      }
      else {
        // we have to resort to elliptical arcs
        context.ellipse( highX, lowY, arcw, arch, 0, -Math.PI / 2, 0, false );
        context.ellipse( highX, highY, arcw, arch, 0, 0, Math.PI / 2, false );
        context.ellipse( lowX, highY, arcw, arch, 0, Math.PI / 2, Math.PI, false );
        context.ellipse( lowX, lowY, arcw, arch, 0, Math.PI, Math.PI * 3 / 2, false );
      }
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
    else {
      // TODO: how to handle fill/stroke delay optimizations here?
      if ( node.hasFill() ) {
        // If we need the fill pattern/gradient to have a different transformation, we can't use fillRect.
        // See https://github.com/phetsims/scenery/issues/543
        if ( node.getFillValue().transformMatrix ) {
          this.writeRectangularPath( context, node );
          node.beforeCanvasFill( wrapper ); // defined in Paintable
          context.fill();
          node.afterCanvasFill( wrapper ); // defined in Paintable
        }
        else {
          node.beforeCanvasFill( wrapper ); // defined in Paintable
          context.fillRect( node._rectX, node._rectY, node._rectWidth, node._rectHeight );
          node.afterCanvasFill( wrapper ); // defined in Paintable
        }
      }
      if ( node.hasPaintableStroke() ) {
        // If we need the fill pattern/gradient to have a different transformation, we can't use fillRect.
        // See https://github.com/phetsims/scenery/issues/543
        if ( node.getStrokeValue().transformMatrix ) {
          this.writeRectangularPath( context, node );
          node.beforeCanvasStroke( wrapper ); // defined in Paintable
          context.stroke();
          node.afterCanvasStroke( wrapper ); // defined in Paintable
        }
        else {
          node.beforeCanvasStroke( wrapper ); // defined in Paintable
          context.strokeRect( node._rectX, node._rectY, node._rectWidth, node._rectHeight );
          node.afterCanvasStroke( wrapper ); // defined in Paintable
        }
      }
    }
  }

  /**
   * @public
   */
  markDirtyRectangle() {
    this.markPaintDirty();
  }

  /**
   * @public
   */
  markDirtyX() {
    this.markDirtyRectangle();
  }

  /**
   * @public
   */
  markDirtyY() {
    this.markDirtyRectangle();
  }

  /**
   * @public
   */
  markDirtyWidth() {
    this.markDirtyRectangle();
  }

  /**
   * @public
   */
  markDirtyHeight() {
    this.markDirtyRectangle();
  }

  /**
   * @public
   */
  markDirtyCornerXRadius() {
    this.markDirtyRectangle();
  }

  /**
   * @public
   */
  markDirtyCornerYRadius() {
    this.markDirtyRectangle();
  }
}

scenery.register( 'RectangleCanvasDrawable', RectangleCanvasDrawable );

Poolable.mixInto( RectangleCanvasDrawable );

export default RectangleCanvasDrawable;