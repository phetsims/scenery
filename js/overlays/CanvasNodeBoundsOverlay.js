// Copyright 2013-2020, University of Colorado Boulder

/**
 * Displays CanvasNode bounds.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import inherit from '../../../phet-core/js/inherit.js';
import CanvasNode from '../nodes/CanvasNode.js';
import scenery from '../scenery.js';
import Trail from'../util/Trail.js';
import ShapeBasedOverlay from './ShapeBasedOverlay.js';

function CanvasNodeBoundsOverlay( display, rootNode ) {
  ShapeBasedOverlay.call( this, display, rootNode, 'canvasNodeBoundsOverlay' );
}

scenery.register( 'CanvasNodeBoundsOverlay', CanvasNodeBoundsOverlay );

inherit( ShapeBasedOverlay, CanvasNodeBoundsOverlay, {
  // @override
  addShapes: function() {
    const self = this;

    new Trail( this.rootNode ).eachTrailUnder( function( trail ) {
      const node = trail.lastNode();
      if ( !node.isVisible() ) {
        // skip this subtree if the node is invisible
        return true;
      }
      if ( ( node instanceof CanvasNode ) && trail.isVisible() ) {
        const transform = trail.getTransform();

        self.addShape( transform.transformShape( Shape.bounds( node.selfBounds ) ), 'rgba(0,255,0,0.8)', true );
      }
    } );
  }
} );

export default CanvasNodeBoundsOverlay;