// Copyright 2013-2024, University of Colorado Boulder

/**
 * Displays CanvasNode bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Shape } from '../../../kite/js/imports.js';
import CanvasNode from '../nodes/CanvasNode.js';
import Display from '../display/Display.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import ShapeBasedOverlay from '../overlays/ShapeBasedOverlay.js';
import type TOverlay from '../overlays/TOverlay.js';
import Trail from '../util/Trail.js';

export default class CanvasNodeBoundsOverlay extends ShapeBasedOverlay implements TOverlay {
  public constructor( display: Display, rootNode: Node ) {
    super( display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  public addShapes(): void {
    new Trail( this.rootNode ).eachTrailUnder( trail => {
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