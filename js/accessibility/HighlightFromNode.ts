// Copyright 2017-2025, University of Colorado Boulder

/**
 * A HighlightPath subtype that is based around a Node. The focusHighlight is constructed based on the bounds of
 * the node. The focusHighlight will update as the Node's bounds changes. Handles transformations so that when the
 * source node is transformed, the HighlightFromNode will be updated as well.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TProperty from '../../../axon/js/TProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Shape from '../../../kite/js/Shape.js';
import optionize from '../../../phet-core/js/optionize.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import type { HighlightPathOptions } from '../accessibility/HighlightPath.js';
import HighlightPath from '../accessibility/HighlightPath.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Trail from '../util/Trail.js';

type SelfOptions = {

  // if true, highlight will surround local bounds instead of parent bounds
  useLocalBounds?: boolean;

  // default value is function of node transform (minus translation), but can be set explicitly.
  // see HighlightPath.getDilationCoefficient(). A number here refers to the amount in global coordinates to
  // dilate the focus highlight.
  dilationCoefficient?: number | null;

  // if true, dilation for bounds around node will increase, see setShapeFromNode()
  useGroupDilation?: boolean;
};

// The transformSourceNode for this highlight will be the provided Node.
export type HighlightFromNodeOptions = SelfOptions & StrictOmit<HighlightPathOptions, 'transformSourceNode'>;

class HighlightFromNode extends HighlightPath {

  // See options for documentation.
  private readonly useLocalBounds: boolean;
  private readonly useGroupDilation: boolean;
  private readonly dilationCoefficient: number | null;

  // Property for a Node's bounds which are currently being observed with the boundsListener. Referenced so that
  // we can remove the listener later.
  private observedBoundsProperty: null | TProperty<Bounds2> = null;

  // Listener that sets the shape of this highlight when the Node bounds change. Referenced so it can be removed later.
  private boundsListener: null | ( ( bounds: Bounds2 ) => void ) = null;

  public constructor( node: Node | null, providedOptions?: HighlightFromNodeOptions ) {

    const options = optionize<HighlightFromNodeOptions, SelfOptions, HighlightPathOptions>()( {
      useLocalBounds: true,
      dilationCoefficient: null,
      useGroupDilation: false
    }, providedOptions );

    options.transformSourceNode = node;

    super( null, options );

    this.useLocalBounds = options.useLocalBounds;
    this.useGroupDilation = options.useGroupDilation;
    this.dilationCoefficient = options.dilationCoefficient;

    if ( node ) {
      this.setShapeFromNode( node );
    }
  }

  /**
   * Update the focusHighlight shape on the path given the node passed in. Depending on options supplied to this
   * HighlightFromNode, the shape will surround the node's bounds or its local bounds, dilated by an amount
   * that is dependent on whether or not this highlight is for group content or for the node itself. See
   * ParallelDOM.setGroupFocusHighlight() for more information on group highlights.
   *
   * node - The Node with a highlight to surround.
   * [trail] - A Trail to use to describe the Node in the global coordinate frame.
   *           Provided by the HighlightOverlay, to support DAG.
   */
  public setShapeFromNode( node: Node, trail?: Trail ): void {

    // cleanup the previous listener
    if ( this.observedBoundsProperty ) {
      assert && assert( this.boundsListener, 'should be a listener if there is a previous focusHighlightNode' );
      this.observedBoundsProperty.unlink( this.boundsListener! );
    }

    // The HighlightOverlay updates highlight positioning with a TransformTracker so the local bounds accurately
    // describe the highlight shape. NOTE: This does not update with changes to visible bounds - scenery
    // does not have support for that at this time (requires a visibleBoundsProperty).
    this.observedBoundsProperty = node.localBoundsProperty;
    this.boundsListener = localBounds => {

      // Ignore setting the shape if we don't yet have finite bounds.
      if ( !localBounds.isFinite() ) {
        return;
      }

      let dilationCoefficient = this.dilationCoefficient;

      // Get the matrix that will transform the node's local bounds to global coordinates.
      // Then apply a pan/zoom correction so that the highlight looks appropriately
      // sized from pan/zoom transformation but other transformations are not applied.
      assert && assert( trail || node.getTrails().length < 2, 'HighlightFromNode requires a unique Trail if using DAG.' );
      const trailToUse = trail || node.getUniqueTrail();
      const matrix = trailToUse.getMatrix()
        .timesMatrix( HighlightPath.getCorrectiveScalingMatrix() );

      // Figure out how much dilation to apply to the focus highlight around the node, calculated unless specified
      // with options
      if ( this.dilationCoefficient === null ) {
        dilationCoefficient = ( this.useGroupDilation ? HighlightPath.getGroupDilationCoefficient( matrix ) :
                                HighlightPath.getDilationCoefficient( matrix ) );
      }

      const visibleBounds = this.useLocalBounds ? node.getVisibleLocalBounds() : node.getVisibleBounds();
      assert && assert( visibleBounds.isFinite(), 'node must have finite bounds.' );
      const dilatedVisibleBounds = visibleBounds.dilated( dilationCoefficient! );

      // Update the line width of the focus highlight based on the transform of the node
      this.setShape( Shape.bounds( dilatedVisibleBounds ) );
    };
    this.observedBoundsProperty.link( this.boundsListener );
  }

  /**
   * Remove the listener from the observedBoundsProperty (which belongs to a provided Node).
   */
  public override dispose(): void {
    if ( this.observedBoundsProperty ) {
      assert && assert( this.boundsListener, 'should be a listener if there is a previous focusHighlightNode' );
      this.observedBoundsProperty.unlink( this.boundsListener! );
    }

    super.dispose();
  }
}

scenery.register( 'HighlightFromNode', HighlightFromNode );

export default HighlightFromNode;