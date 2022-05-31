// Copyright 2022, University of Colorado Boulder

/**
 * Extends LayoutProxyProperty with an added transform tracking callback to notify when the transform changes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { LayoutProxyProperty, Node, scenery, TransformTracker } from '../imports.js';

export default class TrackingLayoutProxyProperty extends LayoutProxyProperty {

  private transformTracker: TransformTracker | null = null;

  constructor( rootNode: Node, leafNode: Node, transformChanged: () => void ) {
    super( rootNode, leafNode );

    this.link( proxy => {
      if ( this.transformTracker ) {
        this.transformTracker.dispose();
        this.transformTracker = null;
      }
      if ( proxy ) {
        this.transformTracker = new TransformTracker( proxy.trail!.copy().addAncestor( rootNode ) );
        this.transformTracker.addListener( transformChanged );
      }
    } );
  }

  /**
   * Releases references
   */
  override dispose(): void {
    this.transformTracker && this.transformTracker.dispose();

    super.dispose();
  }
}

scenery.register( 'TrackingLayoutProxyProperty', TrackingLayoutProxyProperty );
