// Copyright 2023, University of Colorado Boulder

/**
 * Mixin for RichText elements in the hierarchy that should be pooled with a clean() method.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Matrix3 from '../../../../dot/js/Matrix3.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import memoize from '../../../../phet-core/js/memoize.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import { Node, scenery } from '../../imports.js';

const RichTextCleanable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Paintable' );

  return class RichTextCleanableMixin extends type {
    public get isCleanable(): boolean {
      return true;
    }

    /**
     * Releases references
     */
    public clean(): void {
      const thisNode = this as unknown as RichTextCleanableNode;

      // Remove all children (and recursively clean)
      for ( let i = thisNode._children.length - 1; i >= 0; i-- ) {
        const child = thisNode._children[ i ] as RichTextCleanableNode;

        if ( child.isCleanable ) {
          thisNode.removeChild( child );
          child.clean();
        }
      }

      thisNode.matrix = Matrix3.IDENTITY;

      thisNode.freeToPool();
    }
  };
} );
export type RichTextCleanableNode = Node & { clean: () => void; isCleanable: boolean; freeToPool: () => void };

scenery.register( 'RichTextCleanable', RichTextCleanable );

export default RichTextCleanable;
