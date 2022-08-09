// Copyright 2013-2022, University of Colorado Boulder

/**
 * Displays mouse and touch areas when they are customized. Expensive to display!
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import { Shape } from '../../../kite/js/imports.js';
import { Display, TOverlay, Node, scenery, ShapeBasedOverlay, Trail } from '../imports.js';

export default class PointerAreaOverlay extends ShapeBasedOverlay implements TOverlay {
  public constructor( display: Display, rootNode: Node ) {
    super( display, rootNode, 'mouseTouchAreaOverlay' );
  }

  public addShapes(): void {
    new Trail( this.rootNode ).eachTrailUnder( trail => {
      const node = trail.lastNode();
      if ( !node.isVisible() ) {
        // skip this subtree if the node is invisible
        return true;
      }
      if ( ( node.mouseArea || node.touchArea ) && trail.isVisible() ) {
        const transform = trail.getTransform();

        if ( node.mouseArea ) {
          this.addShape( transform.transformShape( node.mouseArea instanceof Bounds2 ? Shape.bounds( node.mouseArea ) : node.mouseArea ), 'rgba(0,0,255,0.8)', true );
        }
        if ( node.touchArea ) {
          this.addShape( transform.transformShape( node.touchArea instanceof Bounds2 ? Shape.bounds( node.touchArea ) : node.touchArea ), 'rgba(255,0,0,0.8)', false );
        }
      }
      return false;
    } );
  }
}

scenery.register( 'PointerAreaOverlay', PointerAreaOverlay );
