// Copyright 2022, University of Colorado Boulder

/**
 * A trait for subtypes of Node, used to prevent children being added/removed to that subtype of Node.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Node, scenery } from '../imports.js';
import memoize from '../../../phet-core/js/memoize.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

const Leaf = memoize( <SuperType extends Constructor<Node>>( type: SuperType ) => {

  return class LeafMixin extends type {
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    public override insertChild( index: number, node: Node ): this {
      throw new Error( 'Attempt to insert child into Leaf' );
    }

    public override removeChildWithIndex( node: Node, indexOfChild: number ): void {
      throw new Error( 'Attempt to remove child from Leaf' );
    }
  };
} );

scenery.register( 'Leaf', Leaf );

export default Leaf;
