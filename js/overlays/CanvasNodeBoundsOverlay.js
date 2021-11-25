// Copyright 2013-2021, University of Colorado Boulder

/**
 * Displays CanvasNode bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import { scenery, Trail, CanvasNode, ShapeBasedOverlay } from '../imports.js';

class CanvasNodeBoundsOverlay extends ShapeBasedOverlay {
  /**
   * @param {Display} display
   * @param {Node} rootNode
   */
  constructor( display, rootNode ) {
    super( display, rootNode, 'canvasNodeBoundsOverlay' );
  }

  /**
   * @public
   * @override
   */
  addShapes() {
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
export default CanvasNodeBoundsOverlay;