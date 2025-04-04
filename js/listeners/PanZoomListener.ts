// Copyright 2017-2025, University of Colorado Boulder

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
import optionize from '../../../phet-core/js/optionize.js';
import ModelViewTransform2 from '../../../phetcommon/js/view/ModelViewTransform2.js';
import isSettingPhetioStateProperty from '../../../tandem/js/isSettingPhetioStateProperty.js';
import MultiListener from '../listeners/MultiListener.js';
import type { MultiListenerOptions } from '../listeners/MultiListener.js';
import MultiListenerPress from '../listeners/MultiListenerPress.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';

// constants
// Reusable Matrix3 instance to avoid creating lots of them
const SCRATCH_MATRIX = new Matrix3();

type SelfOptions = {

  // these bounds should be fully filled with content at all times, in the global coordinate frame
  panBounds?: Bounds2;

  // Bounds for the target node that get transformed with this listener and fill panBounds,
  // useful if the targetNode bounds do not accurately describe the targetNode (like if invisible content
  // extends off screen). Defaults to targetNode bounds if null. Bounds in the global coordinate frame of the
  // target Node.
  targetBounds?: Bounds2 | null;

  // Scale that accurately describes scale of the targetNode, but is different from the actual scale of the
  // targetNode's transform. This scale is applied to translation Vectors for the TargetNode during panning. If
  // targetNode children get scaled uniformly (such as in response to window resizing or native browser zoom), you
  // likely want that scale to be applied during translation operations so that pan/zoom behaves
  // the same regardless of window size or native browser zoom.
  targetScale?: number;
};
export type PanZoomListenerOptions = SelfOptions & MultiListenerOptions;

class PanZoomListener extends MultiListener {

  protected _panBounds: Bounds2;
  protected _targetBounds: Bounds2;
  protected _targetScale: number;

  // Only needed for PhET-iO instrumented. The pan bounds of the source so if the destination bounds are different due
  // to a differently sized iframe or window, this can be used to determine a correction for the destination
  // targetNode transform. This could be removed by work recommended in
  protected sourceFramePanBoundsProperty: Property<Bounds2>;

  /**
   * @param targetNode - The Node that should be transformed by this PanZoomListener.
   * @param [providedOptions].
   */
  public constructor( targetNode: Node, providedOptions?: PanZoomListenerOptions ) {

    const options = optionize<PanZoomListenerOptions, SelfOptions, PanZoomListenerOptions>()( {
      panBounds: Bounds2.NOTHING,
      targetBounds: null,
      targetScale: 1,

      // by default, the PanZoomListener does now allow rotation
      allowRotation: false
    }, providedOptions );

    super( targetNode, options );

    this._panBounds = options.panBounds;
    this._targetBounds = options.targetBounds || targetNode.globalBounds.copy();
    this._targetScale = options.targetScale;

    this.sourceFramePanBoundsProperty = new Property( this._panBounds, {
      tandem: options.tandem?.createTandem( 'sourceFramePanBoundsProperty' ),
      phetioReadOnly: true,
      phetioValueType: Bounds2.Bounds2IO
    } );

    this.sourceFramePanBoundsProperty.lazyLink( () => {
      if ( isSettingPhetioStateProperty.value ) {

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
   */
  protected correctReposition(): void {

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
   */
  protected override addPress( press: MultiListenerPress ): void {
    super.addPress( press );

    // don't show the pressCursor if our bounds are limited by pan bounds, and we cannot pan anywhere
    const transformedBounds = this._targetBounds.transformed( this._targetNode.getMatrix() );
    const boundsLimited = transformedBounds.equalsEpsilon( this._panBounds, 1E-8 );
    press.pointer.cursor = boundsLimited ? null : this._pressCursor;
  }

  /**
   * Reposition but keep content within this._panBounds.
   */
  protected override reposition(): void {
    super.reposition();
    this.correctReposition();
  }

  /**
   * Reset the transform on the targetNode and follow up by making sure that the content is still within panBounds.
   */
  public override resetTransform(): void {
    super.resetTransform();
    this.correctReposition();
  }

  /**
   * Set the containing panBounds and then make sure that the targetBounds fully fill the new panBounds.
   */
  public setPanBounds( panBounds: Bounds2 ): void {
    this._panBounds = panBounds;

    this.sourceFramePanBoundsProperty.set( this._panBounds );
    this.correctReposition();
  }

  /**
   * Set the targetBounds which should totally fill the panBounds at all times. Useful if the targetNode has bounds
   * which don't accurately describe the node. For instance, if an overlay plane is on top of the node and extends
   * beyond the dimensions of the visible node.
   *
   * targetBounds - in the global coordinate frame
   */
  public setTargetBounds( targetBounds: Bounds2 ): void {
    this._targetBounds = targetBounds;
    this.correctReposition();
  }

  /**
   * Set the representative scale of the target Node. If the targetBounds are different from the targetNode.bounds
   * it may be useful to correct changes to panning and zooming by a scale that is different from the
   * actual scale applied to the targetNode during panning.
   */
  public setTargetScale( scale: number ): void {
    this._targetScale = scale;
  }

  /**
   * Get the targetBounds, in the global coordinate frame.
   */
  public getTargetBounds(): Bounds2 {
    return this._targetBounds;
  }
}

scenery.register( 'PanZoomListener', PanZoomListener );
export default PanZoomListener;