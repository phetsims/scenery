// Copyright 2023-2025, University of Colorado Boulder

/**
 * Mixin for RichText elements in the hierarchy that should be pooled with a clean() method.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Matrix3 from '../../../../dot/js/Matrix3.js';
import inheritance from '../../../../phet-core/js/inheritance.js';
import memoize from '../../../../phet-core/js/memoize.js';
import { TPoolable } from '../../../../phet-core/js/Pool.js';
import Constructor from '../../../../phet-core/js/types/Constructor.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';

type TRichTextCleanable = {
  readonly isCleanable: boolean;
  clean(): void;
};

const RichTextCleanable = memoize( <SuperType extends Constructor>( type: SuperType ): SuperType & Constructor<TRichTextCleanable> => {
  assert && assert( _.includes( inheritance( type ), Node ), 'Only Node subtypes should mix Paintable' );

  return class RichTextCleanableMixin extends type implements TRichTextCleanable {
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
        thisNode.removeChild( child );

        if ( child.isCleanable ) {
          child.clean();
        }
      }

      thisNode.matrix = Matrix3.IDENTITY;

      thisNode.freeToPool();
    }
  };
} );
export type RichTextCleanableNode = Node & TPoolable & TRichTextCleanable;

scenery.register( 'RichTextCleanable', RichTextCleanable );

export default RichTextCleanable;