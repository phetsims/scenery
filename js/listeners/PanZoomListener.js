// Copyright 2017-2023, University of Colorado Boulder

/**
 * A MultiListener that is designed to pan and zoom a target Node, where you can provide limiting and
 * describing bounds so that the targetNode is limited to a region.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Property from '../../../axon/js/Property.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Matrix3 from '../../../dot/js/Matrix3.js';
import merge from '../../../phet-core/js/merge.js';
import ModelViewTransform2 from '../../../phetcommon/js/view/ModelViewTransform2.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { MultiListener, scenery } from '../imports.js';

// constants
// Reusable Matrix3 instance to avoid creating lots of them
const SCRATCH_MATRIX = new Matrix3();

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
      allowRotation: false,

      // {Tandem}
      tandem: Tandem.OPTIONAL
    }, options );

    super( targetNode, options );

    // @private {Bounds2} - see options
    this._panBounds = options.panBounds;
    this._targetBounds = options.targetBounds || targetNode.globalBounds.copy();

    // @protected {number}
    this._targetScale = options.targetScale;

    // @protected {Property.<Bounds2>} - Only needed for PhET-iO instrumented. The pan bounds of the source
    // so if the destination bounds are different due to a differently sized iframe or window,
    // this can be used to determine a correction for the destination targetNode transform.
    // This could be removed by work recommended in

    // When generating a PhET-iO API, the specific bounds of the window should be excluded from the initial state
    // so that the initial state part of the API doesn't depend on the window size.
    this.sourceFramePanBoundsProperty = new Property( Tandem.API_GENERATION ? new Bounds2( 0, 0, 0, 0 ) : this._panBounds, {
      tandem: options.tandem.createTandem( 'sourceFramePanBoundsProperty' ),
      phetioReadOnly: true,
      phetioValueType: Bounds2.Bounds2IO
    } );

    this.sourceFramePanBoundsProperty.lazyLink( () => {
      const simGlobal = _.get( window, 'phet.joist.sim', null ); // returns null if global isn't found

      if ( ( simGlobal && simGlobal.isSettingPhetioStateProperty.value ) ) {

        // The matrixProperty has transformations relative to the global view coordinates of the source simulation,
        // so it will not be correct if source and destination frames are different sizes. This will map transforamtions
        // if destination frame has different size.
        const sourceDestinationTransform = ModelViewTransform2.createRectangleMapping( this.sourceFramePanBoundsProperty.get(), this._panBounds );

        const newTranslation = this._targetNode.matrix.translation.componentMultiply( sourceDestinationTransform.matrix.getScaleVector() );
        const scale = this.matrixProperty.get().getScaleVector();
        this.matrixProperty.set( Matrix3.translationFromVector( newTranslation ).timesMatrix( Matrix3.scaling( scale.x, scale.y ) ) );
      }
    }, {

      // so that the listener will be called only after the matrixProperty is up to date in the downstream sim
      phetioDependencies: [ this.matrixProperty ]
    } );
  }

  /**
   * If the targetNode is larger than the panBounds specified, keep the panBounds completely filled with
   * targetNode content.
   *
   * @protected
   */
  correctReposition() {

    // Save values of the current matrix, so that we only do certain work when the matrix actually changes
    SCRATCH_MATRIX.set( this._targetNode.matrix );

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

    // Update Property with matrix once position has been corrected to notify listeners and set PhET-iO state, but
    // only notify when there has been an actual change.
    if ( !SCRATCH_MATRIX.equals( this._targetNode.matrix ) ) {
      this.matrixProperty.set( this._targetNode.matrix.copy() );
    }
  }

  /**
   * If the transformed targetBounds are equal to the panBounds, there is no space for us to pan so do not change
   * the pointer cursor.
   * @protected
   * @override
   *
   * @param {Press} press
   */
  addPress( press ) {
    super.addPress( press );

    // don't show the pressCursor if our bounds are limited by pan bounds, and we cannot pan anywhere
    const transformedBounds = this._targetBounds.transformed( this._targetNode.getMatrix() );
    const boundsLimited = transformedBounds.equalsEpsilon( this._panBounds, 1E-8 );
    press.pointer.cursor = boundsLimited ? null : this._pressCursor;
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

    // When generating a PhET-iO API, the specific bounds of the window should be excluded from the initial state
    // so that the initial state part of the API doesn't depend on the window size.
    if ( !Tandem.API_GENERATION ) {
      this.sourceFramePanBoundsProperty.set( this._panBounds );
    }
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