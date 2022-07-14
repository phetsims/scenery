// Copyright 2022, University of Colorado Boulder

/**
 * A trait for subtypes of Node, used to prevent children being added/removed to that subtype of Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inheritance from '../../../phet-core/js/inheritance.js';
import { scenery, Node } from '../imports.js';
import memoize from '../../../phet-core/js/memoize.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

const Leaf = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Leaf' );

  return class LeafMixin extends type {
    public constructor( ...args: any[] ) {
      super( ...args );
    }

    public insertChild( index: number, node: Node ): void {
      throw new Error( 'Attempt to insert child into Leaf' );
    }

    public removeChildWithIndex( node: Node, indexOfChild: number ): void {
      throw new Error( 'Attempt to remove child from Leaf' );
    }
  };
} );

scenery.register( 'Leaf', Leaf );

export default Leaf;
