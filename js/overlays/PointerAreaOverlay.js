// Copyright 2013-2020, University of Colorado Boulder

/**
 * Displays mouse and touch areas when they are customized. Expensive to display!
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';
import ShapeBasedOverlay from './ShapeBasedOverlay.js';

class PointerAreaOverlay extends ShapeBasedOverlay {
  /**
   * @param {Display} display
   * @param {Node} rootNode
   */
  constructor( display, rootNode ) {
    super( display, rootNode, 'mouseTouchAreaOverlay' );
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
      if ( ( node.mouseArea || node.touchArea ) && trail.isVisible() ) {
        const transform = trail.getTransform();

        if ( node.mouseArea ) {
          this.addShape( transform.transformShape( node.mouseArea.isBounds ? Shape.bounds( node.mouseArea ) : node.mouseArea ), 'rgba(0,0,255,0.8)', true );
        }
        if ( node.touchArea ) {
          this.addShape( transform.transformShape( node.touchArea.isBounds ? Shape.bounds( node.touchArea ) : node.touchArea ), 'rgba(255,0,0,0.8)', false );
        }
      }
    } );
  }
}

scenery.register( 'PointerAreaOverlay', PointerAreaOverlay );
export default PointerAreaOverlay;