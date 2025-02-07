// Copyright 2013-2025, University of Colorado Boulder

/**
 * Displays CanvasNode bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import type Display from '../display/Display.js';
import CanvasNode from '../nodes/CanvasNode.js';
import type Node from '../nodes/Node.js';
import ShapeBasedOverlay from '../overlays/ShapeBasedOverlay.js';
import type TOverlay from '../overlays/TOverlay.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';
import TrailPointer from '../util/TrailPointer.js';

export default class CanvasNodeBoundsOverlay extends ShapeBasedOverlay implements TOverlay {
  public constructor( display: Display, rootNode: Node ) {
    super( display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  public addShapes(): void {
    TrailPointer.eachTrailUnder( new Trail( this.rootNode ), trail => {
      const node = trail.lastNode();
      if ( !node.isVisible() ) {
        // skip this subtree if the node is invisible
        return true;
      }
      if ( ( node instanceof CanvasNode ) && trail.isVisible() ) {
        const transform = trail.getTransform();

        this.addShape( transform.transformShape( Shape.bounds( node.selfBounds ) ), 'rgba(0,255,0,0.8)', true );
      }
      return false;
    } );
  }
}

scenery.register( 'CanvasNodeBoundsOverlay', CanvasNodeBoundsOverlay );