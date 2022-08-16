// Copyright 2017-2022, University of Colorado Boulder

/**
 * A FocusHighlightPath subtype that is based around a Node. The focusHighlight is constructed based on the bounds of
 * the node. The focusHighlight will update as the Node's bounds changes. Handles transformations so that when the
 * source node is transformed, the FocusHighlightFromNode will
 * updated be as well.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import TProperty from '../../../axon/js/TProperty.js';
import StrictOmit from '../../../phet-core/js/types/StrictOmit.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import { Shape } from '../../../kite/js/imports.js';
import optionize from '../../../phet-core/js/optionize.js';
import { FocusHighlightPath, Node, scenery } from '../imports.js';
import { FocusHighlightPathOptions } from './FocusHighlightPath.js';

type SelfOptions = {

  // if true, highlight will surround local bounds instead of parent bounds
  useLocalBounds?: boolean;

  // default value is function of node transform (minus translation), but can be set explicitly.
  // see FocusHighlightPath.getDilationCoefficient(). A number here refers to the amount in global coordinates to
  // dilate the focus highlight.
  dilationCoefficient?: number | null;

  // if true, dilation for bounds around node will increase, see setShapeFromNode()
  useGroupDilation?: boolean;
};

// The transformSourceNode for this highlight will be the provided Node.
export type FocusHighlightFromNodeOptions = SelfOptions & StrictOmit<FocusHighlightPathOptions, 'transformSourceNode'>;

class FocusHighlightFromNode extends FocusHighlightPath {

  // See options for documentation.
  private readonly useLocalBounds: boolean;
  private readonly useGroupDilation: boolean;
  private readonly dilationCoefficient: number | null;

  // Property for a Node's bounds which are currently being observed with the boundsListener. Referenced so that
  // we can remove the listener later.
  private observedBoundsProperty: null | TProperty<Bounds2> = null;

  // Listener that sets the shape of this highlight when the Node bounds change. Referenced so it can be removed later.
  private boundsListener: null | ( ( bounds: Bounds2 ) => void ) = null;

  public constructor( node: Node | null, providedOptions?: FocusHighlightFromNodeOptions ) {

    const options = optionize<FocusHighlightFromNodeOptions, SelfOptions, FocusHighlightPathOptions>()( {
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
   * FocusHighlightFromNode, the shape will surround the node's bounds or its local bounds, dilated by an amount
   * that is dependent on whether or not this highlight is for group content or for the node itself. See
   * ParallelDOM.setGroupFocusHighlight() for more information on group highlights.
   */
  public setShapeFromNode( node: Node ): void {

    // cleanup the previous listener
    if ( this.observedBoundsProperty ) {
      assert && assert( this.boundsListener, 'should be a listener if there is a previous focusHighlightNode' );
      this.observedBoundsProperty.unlink( this.boundsListener! );
    }

    this.observedBoundsProperty = this.useLocalBounds ? node.localBoundsProperty : node.boundsProperty;

    this.boundsListener = bounds => {

      // Ignore setting the shape if we don't yet have finite bounds.
      if ( !bounds.isFinite() ) {
        return;
      }

      let dilationCoefficient = this.dilationCoefficient;

      // Figure out how much dilation to apply to the focus highlight around the node, calculated unless specified
      // with options
      if ( this.dilationCoefficient === null ) {
        dilationCoefficient = ( this.useGroupDilation ? FocusHighlightPath.getGroupDilationCoefficient( node ) :
                                FocusHighlightPath.getDilationCoefficient( node ) );
      }
      const dilatedBounds = bounds.dilated( dilationCoefficient! );

      // Update the line width of the focus highlight based on the transform of the node
      this.updateLineWidthFromNode( node );
      this.setShape( Shape.bounds( dilatedBounds ) );
    };
    this.observedBoundsProperty.link( this.boundsListener );
  }

  /**
   * Update the line width of both inner and outer highlights based on transform of the Node.
   */
  private updateLineWidthFromNode( node: Node ): void {

    // Note that lineWidths provided by options can override width determined from Node transform.
    this.lineWidth = this.getOuterLineWidth( node );
    this.innerHighlightPath.lineWidth = this.getInnerLineWidth( node );
  }
}

scenery.register( 'FocusHighlightFromNode', FocusHighlightFromNode );

export default FocusHighlightFromNode;