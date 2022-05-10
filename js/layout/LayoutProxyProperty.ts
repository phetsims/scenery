// Copyright 2022, University of Colorado Boulder

/**
 * A Property that takes the value of:
 * - a LayoutProxy with the single connected Trail (if it exists)
 * - null if there are zero or 2+ connected Trails between the two Nodes
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import DerivedProperty from '../../../axon/js/DerivedProperty.js';
import Tandem from '../../../tandem/js/Tandem.js';
import { LayoutProxy, Node, scenery, Trail, TrailsBetweenProperty } from '../imports.js';

export default class LayoutProxyProperty extends DerivedProperty<LayoutProxy | null, [ Trail[] ]> {

  private readonly trailsBetweenProperty: TrailsBetweenProperty

  constructor( rootNode: Node, leafNode: Node ) {

    const trailsBetweenProperty = new TrailsBetweenProperty( rootNode, leafNode );

    super( [ trailsBetweenProperty ], trails => {
      return trails.length === 1 ? LayoutProxy.pool.create( trails[ 0 ].copy().removeAncestor() ) : null;
    }, {
      tandem: Tandem.OPT_OUT
    } );

    this.trailsBetweenProperty = trailsBetweenProperty;
    this.lazyLink( ( value, oldValue ) => {
      oldValue && oldValue.dispose();
    } );
  }

  override dispose(): void {
    this.trailsBetweenProperty.dispose();

    super.dispose();
  }
}

scenery.register( 'LayoutProxyProperty', LayoutProxyProperty );
