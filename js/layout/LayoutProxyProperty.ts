// Copyright 2022, University of Colorado Boulder

/**
 * A Property that takes the value of:
 * - a LayoutProxy with the single connected Trail (if it exists)
 * - null if there are zero or 2+ connected Trails between the two Nodes
 *
 * When defined, this will provide a LayoutProxy for the leafNode within the rootNode's local coordinate frame. This
 * will allow positioning the leafNode within the rootNode's coordinate frame (which is ONLY well-defined when there
 * is exactly one trail between the two).
 *
 * Thus, it will only be defined as a proxy if there is a unique trail between the two Nodes. This is needed for layout
 * work, where often we'll need to provide a proxy IF this condition is true, and NO proxy if it's not (since layout
 * would be ambiguous). E.g. for ManualConstraint, if a Node isn't connected to the root, there's nothing the constraint
 * can do.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { DerivedProperty1 } from '../../../axon/js/DerivedProperty.js';
import { LayoutProxy, Node, scenery, Trail, TrailsBetweenProperty, TransformTracker } from '../imports.js';

type SelfOptions = {
  // If provided, this will be called when the transform of the proxy changes
  onTransformChange?: () => void;
};

export type LayoutProxyPropertyOptions = SelfOptions;

export default class LayoutProxyProperty extends DerivedProperty1<LayoutProxy | null, Trail[]> {

  // This will contain the number of trails connecting our rootNode and leafNode. Our value will be solely based off of
  // this Property's value, and is thus created as a DerivedProperty.
  private readonly trailsBetweenProperty: TrailsBetweenProperty;

  // Should be set if we provide an onTransformChange callback
  private transformTracker: TransformTracker | null = null;

  /**
   * @param rootNode - The root whose local coordinate frame we'll want the proxy to be in
   * @param leafNode - The leaf that we'll create the proxy for
   * @param providedOptions
   */
  public constructor( rootNode: Node, leafNode: Node, providedOptions?: LayoutProxyPropertyOptions ) {

    const trailsBetweenProperty = new TrailsBetweenProperty( rootNode, leafNode );

    super( [ trailsBetweenProperty ], trails => {
      return trails.length === 1 ? LayoutProxy.pool.create( trails[ 0 ].copy().removeAncestor() ) : null;
    } );

    this.trailsBetweenProperty = trailsBetweenProperty;
    this.lazyLink( ( value, oldValue ) => {
      oldValue && oldValue.dispose();
    } );

    const onTransformChange = providedOptions?.onTransformChange;
    if ( onTransformChange ) {
      this.link( proxy => {
        if ( this.transformTracker ) {
          this.transformTracker.dispose();
          this.transformTracker = null;
        }
        if ( proxy ) {
          this.transformTracker = new TransformTracker( proxy.trail!.copy().addAncestor( rootNode ) );
          this.transformTracker.addListener( onTransformChange );
        }
      } );
    }
  }

  public override dispose(): void {
    this.trailsBetweenProperty.dispose();
    this.transformTracker && this.transformTracker.dispose();

    super.dispose();
  }
}

scenery.register( 'LayoutProxyProperty', LayoutProxyProperty );
