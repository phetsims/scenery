// Copyright 2019-2025, University of Colorado Boulder

/**
 * Displays the "hittable" mouse/touch regions for items with input listeners.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import type Display from '../display/Display.js';
import type Node from '../nodes/Node.js';
import ShapeBasedOverlay from '../overlays/ShapeBasedOverlay.js';
import type TOverlay from '../overlays/TOverlay.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';
import TrailPointer from '../util/TrailPointer.js';

export default class HitAreaOverlay extends ShapeBasedOverlay implements TOverlay {
  public constructor( display: Display, rootNode: Node ) {
    super( display, rootNode, 'hitAreaOverlay' );
  }

  public addShapes(): void {
    TrailPointer.eachTrailUnder( new Trail( this.rootNode ), trail => {
      const node = trail.lastNode();

      if ( !node.isVisible() || node.pickable === false ) {
        // skip this subtree if the node is invisible
        return true;
      }

      if ( node.inputListeners.length && trail.isVisible() ) {
        const mouseShape = HitAreaOverlay.getLocalMouseShape( node );
        const touchShape = HitAreaOverlay.getLocalTouchShape( node );
        const matrix = trail.getMatrix();

        if ( !mouseShape.bounds.isEmpty() ) {
          this.addShape( mouseShape.transformed( matrix ), 'rgba(0,0,255,0.8)', true );
        }
        if ( !touchShape.bounds.isEmpty() ) {
          this.addShape( touchShape.transformed( matrix ), 'rgba(255,0,0,0.8)', false );
        }
      }

      return false;
    } );
  }

  private static getLocalMouseShape( node: Node ): Shape {
    let shape = Shape.union( [
      node.mouseArea ? ( node.mouseArea instanceof Shape ? node.mouseArea : Shape.bounds( node.mouseArea ) ) : node.getSelfShape(),
      ...node.children.filter( ( child: Node ) => {
        return node.visible && node.pickable !== false;
      } ).map( child => {
        return HitAreaOverlay.getLocalMouseShape( child ).transformed( child.matrix );
      } )
    ] );
    if ( node.hasClipArea() ) {
      shape = shape.shapeIntersection( node.clipArea! );
    }
    return shape;
  }

  private static getLocalTouchShape( node: Node ): Shape {
    let shape = Shape.union( [
      node.touchArea ? ( node.touchArea instanceof Shape ? node.touchArea : Shape.bounds( node.touchArea ) ) : node.getSelfShape(),
      ...node.children.filter( ( child: Node ) => {
        return node.visible && node.pickable !== false;
      } ).map( child => {
        return HitAreaOverlay.getLocalTouchShape( child ).transformed( child.matrix );
      } )
    ] );
    if ( node.hasClipArea() ) {
      shape = shape.shapeIntersection( node.clipArea! );
    }
    return shape;
  }
}

scenery.register( 'HitAreaOverlay', HitAreaOverlay );