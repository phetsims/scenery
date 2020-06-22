// Copyright 2017-2020, University of Colorado Boulder

/**
 * A MultiListener that is designed to pan and zoom a target Node, where you can provide limiting and
 * describing bounds so that the targetNode is limited to a region.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import MultiListener from './MultiListener.js';

class PanZoomListener extends MultiListener {

  /**
   * @param {Node} targetNode - The Node that should be transformed by this PanZoomListener.
   * @param {Object} [options] - See the constructor body (below) for documented options.
   */
  constructor( targetNode, options ) {

    options = merge( {

      // {Bounds2} - these bounds should be fully filled with content at all times, in the global coordinate frame
      panBounds: Bounds2.NOTHING,

      // {null|Bounds2} - Bounds for the target node that get transformed with this listener and fill panBounds,
      // useful if the targetNode bounds do not accurately describe the targetNode (like if invisible content
      // extends off screen). Defaults to targetNode bounds if null. Bounds in the global coordinate frame of the
      // target Node.
      targetBounds: null,

      // {number} - Scale that accurately describes scale of the targetNode, but is different from the actual
      // scale of the targetNode's transform. This scale is applied to translation Vectors for the TargetNode during
      // panning. If targetNode children get scaled uniformly (such as in response to window resizing or native
      // browser zoom), you likely want that scale to be applied during translation operations so that pan/zoom behaves
      // the same regardless of window size or native browser zoom.
      targetScale: 1,

      // {boolean} - by default, the PanZoomListener does now allow rotation
      allowRotation: false
    }, options );

    super( targetNode, options );

    // @private {Bounds2} - see options
    this._panBounds = options.panBounds;
    this._targetBounds = options.targetBounds || targetNode.globalBounds.copy();

    // @protected {number}
    this._targetScale = options.targetScale;
  }

  /**
   * If the targetNode is larger than the panBounds specified, keep the panBounds completely filled with
   * targetNode content.
   *
   * @protected
   */
  correctReposition() {

    // the targetBounds transformed by the targetNode's transform, to determine if targetBounds are out of panBounds
    const transformedBounds = this._targetBounds.transformed( this._targetNode.getMatrix() );

    // Don't let panning go through if the node is fully contained by the panBounds
    if ( transformedBounds.left > this._panBounds.left ) {
      this._targetNode.left = this._panBounds.left - ( transformedBounds.left - this._targetNode.left );
    }
    if ( transformedBounds.top > this._panBounds.top ) {
      this._targetNode.top = this._panBounds.top - ( transformedBounds.top - this._targetNode.top );
    }
    if ( transformedBounds.right < this._panBounds.right ) {
      this._targetNode.right = this._panBounds.right + ( this._targetNode.right - transformedBounds.right );
    }
    if ( transformedBounds.bottom < this._panBounds.bottom ) {
      this._targetNode.bottom = this._panBounds.bottom + ( this._targetNode.bottom - transformedBounds.bottom );
    }
  }

  /**
   * Reposition but keep content within this._panBounds.
   * @public
   * @override
   */
  reposition() {
    super.reposition();
    this.correctReposition();
  }

  /**
   * Reset the transform on the targetNode and follow up by making sure that the content is still within panBounds.
   * @public
   * @override
   */
  resetTransform() {
    MultiListener.prototype.resetTransform.call( this );
    this.correctReposition();
  }

  /**
   * Set the containing panBounds and then make sure that the targetBounds fully fill the new panBounds.
   * @override
   * @public
   *
   * @param {Bounds2} panBounds
   */
  setPanBounds( panBounds ) {
    this._panBounds = panBounds;
    this.correctReposition();
  }

  /**
   * Set the targetBounds which should totally fill the panBounds at all times. Useful if the targetNode has bounds
   * which don't accurately describe the node. For instance, if an overlay plane is on top of the node and extends
   * beyond the dimensions of the visible node.
   * @public
   *
   * @param {Bounds2} targetBounds - in the global coordinate frame
   */
  setTargetBounds( targetBounds ) {
    this._targetBounds = targetBounds;
    this.correctReposition();
  }

  /**
   * Set the representative scale of the target Node. If the targetBounds are different from the targetNode.bounds
   * it may be useful to correct changes to panning and zooming by a scale that is different from the
   * actual scale applied to the targetNode during panning.
   * @public
   * @param {number} scale
   */
  setTargetScale( scale ) {
    this._targetScale = scale;
  }

  /**
   * Get the targetBounds, in the global coordinate frame.
   * @public
   *
   * @returns {Bounds2}
   */
  getTargetBounds( targetBounds ) {
    return this._targetBounds;
  }
}

scenery.register( 'PanZoomListener', PanZoomListener );
export default PanZoomListener;