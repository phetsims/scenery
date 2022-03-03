// Copyright 2021-2022, University of Colorado Boulder

/**
 * Provides a minimum and preferred width. The minimum width is set by the component, so that layout containers could
 * know how "small" the component can be made. The preferred width is set by the layout container, and the component
 * should adjust its size so that it takes up that width.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import memoize from '../../../phet-core/js/memoize.js';
import { scenery, Node } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

const WIDTH_SIZABLE_OPTION_KEYS = [
  'preferredWidth',
  'minimumWidth'
];

type WidthSizableSelfOptions = {
  preferredWidth?: number | null,
  minimumWidth?: number | null
};

const WidthSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const clazz = class extends type {

    preferredWidthProperty: TinyProperty<number | null>;
    minimumWidthProperty: TinyProperty<number | null>;

    constructor( ...args: any[] ) {
      super( ...args );

      this.preferredWidthProperty = new TinyProperty<number | null>( null );
      this.minimumWidthProperty = new TinyProperty<number | null>( null );
    }

    get preferredWidth(): number | null {
      return this.preferredWidthProperty.value;
    }

    set preferredWidth( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredWidth should be null or a non-negative finite number' );

      this.preferredWidthProperty.value = value;
    }

    get minimumWidth(): number | null {
      return this.minimumWidthProperty.value;
    }

    set minimumWidth( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumWidthProperty.value = value;
    }

    // Detection flag for this trait
    get widthSizable(): boolean { return true; }
  };

  // If we're extending into a Node type, include option keys
  // TODO: This is ugly, we'll need to mutate after construction, no?
  if ( type.prototype._mutatorKeys ) {
    clazz.prototype._mutatorKeys = type.prototype._mutatorKeys.concat( WIDTH_SIZABLE_OPTION_KEYS );
  }

  return clazz;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as widthSizable
const wrapper = () => WidthSizable( Node );
type WidthSizableNode = InstanceType<ReturnType<typeof wrapper>>;
const isWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.widthSizable;
};

scenery.register( 'WidthSizable', WidthSizable );
export default WidthSizable;
export { isWidthSizable };
export type { WidthSizableNode, WidthSizableSelfOptions };
