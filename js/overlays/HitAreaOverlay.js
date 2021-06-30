// Copyright 2019-2021, University of Colorado Boulder

/**
 * Displays the "hittable" mouse/touch regions for items with input listeners.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Shape from '../../../kite/js/Shape.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';
import ShapeBasedOverlay from './ShapeBasedOverlay.js';

class HitAreaOverlay extends ShapeBasedOverlay {
  /**
   * @param {Display} display
   * @param {Node} rootNode
   */
  constructor( display, rootNode ) {
    super( display, rootNode, 'hitAreaOverlay' );
  }

  /**
   * Adds shapes
   * @protected
   * @override
   */
  addShapes() {
    new Trail( this.rootNode ).eachTrailUnder( trail => {
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

  /**
   * @private
   *
   * @param {Node} node
   * @returns {Shape}
   */
  static getLocalMouseShape( node ) {
    let shape = Shape.union( [
      node.mouseArea ? ( node.mouseArea instanceof Shape ? node.mouseArea : Shape.bounds( node.mouseArea ) ) : node.getSelfShape(),
      ...node.children.filter( child => {
        return node.visible && node.pickable !== false;
      } ).map( child => {
        return HitAreaOverlay.getLocalMouseShape( child ).transformed( child.matrix );
      } )
    ] );
    if ( node.hasClipArea() ) {
      shape = shape.shapeIntersection( node.clipArea );
    }
    return shape;
  }

  /**
   * @private
   *
   * @param {Node} node
   * @returns {Shape}
   */
  static getLocalTouchShape( node ) {
    let shape = Shape.union( [
      node.touchArea ? ( node.touchArea instanceof Shape ? node.touchArea : Shape.bounds( node.touchArea ) ) : node.getSelfShape(),
      ...node.children.filter( child => {
        return node.visible && node.pickable !== false;
      } ).map( child => {
        return HitAreaOverlay.getLocalTouchShape( child ).transformed( child.matrix );
      } )
    ] );
    if ( node.hasClipArea() ) {
      shape = shape.shapeIntersection( node.clipArea );
    }
    return shape;
  }
}

scenery.register( 'HitAreaOverlay', HitAreaOverlay );
export default HitAreaOverlay;