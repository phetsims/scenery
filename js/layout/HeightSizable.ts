// Copyright 2021-2022, University of Colorado Boulder

/**
 * Provides a minimum and preferred height. The minimum height is set by the component, so that layout containers could
 * know how "small" the component can be made. The preferred height is set by the layout container, and the component
 * should adjust its size so that it takes up that height.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import memoize from '../../../phet-core/js/memoize.js';
import { scenery, Node } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

const HEIGHT_SIZABLE_OPTION_KEYS = [
  'preferredHeight',
  'minimumHeight'
];

type HeightSizableSelfOptions = {
  preferredHeight: number | null,
  minimumHeight: number | null
};

const HeightSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const clazz = class extends type {

    preferredHeightProperty: TinyProperty<number | null>;
    minimumHeightProperty: TinyProperty<number | null>;

    constructor( ...args: any[] ) {
      super( ...args );

      this.preferredHeightProperty = new TinyProperty<number | null>( null );
      this.minimumHeightProperty = new TinyProperty<number | null >( null );
    }

    get preferredHeight(): number | null {
      return this.preferredHeightProperty.value;
    }

    set preferredHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredHeight should be null or a non-negative finite number' );

      this.preferredHeightProperty.value = value;
    }

    get minimumHeight(): number | null {
      return this.minimumHeightProperty.value;
    }

    set minimumHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumHeightProperty.value = value;
    }

    // Detection flag for this trait
    get heightSizable(): boolean { return true; }
  };

  // If we're extending into a Node type, include option keys
  // TODO: This is ugly, we'll need to mutate after construction, no?
  if ( type.prototype._mutatorKeys ) {
    clazz.prototype._mutatorKeys = type.prototype._mutatorKeys.concat( HEIGHT_SIZABLE_OPTION_KEYS );
  }

  return clazz;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as HeightSizable
const wrapper = () => HeightSizable( Node );
type HeightSizableNode = InstanceType<ReturnType<typeof wrapper>>;
const isHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.heightSizable;
};

scenery.register( 'HeightSizable', HeightSizable );
export default HeightSizable;
export { isHeightSizable };
export type { HeightSizableNode, HeightSizableSelfOptions };
