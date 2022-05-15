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
import { scenery, Node, REQUIRES_BOUNDS_OPTION_KEYS } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

export const HEIGHT_SIZABLE_OPTION_KEYS = [
  'preferredHeight',
  'minimumHeight',
  'localPreferredHeight',
  'localMinimumHeight',
  'heightSizable'
];

export type HeightSizableSelfOptions = {
  // Sets the preferred height of the Node in the parent coordinate frame. Nodes that implement this will attempt to keep
  // their `node.height` at this value. If null, the node will likely set its configuration to the minimum height.
  // NOTE: changing this or localPreferredHeight will adjust the other.
  // NOTE: preferredHeight is not guaranteed currently. The component may end up having a smaller or larger size
  preferredHeight?: number | null;

  // Sets the minimum height of the Node in the parent coordinate frame. Usually not directly set by a client.
  // Usually a resizable Node will set its localMinimumHeight (and that will get transferred to this value in the
  // parent coordinate frame).
  // NOTE: changing this or localMinimumHeight will adjust the other.
  // NOTE: when the Node's transform is updated, this value is recomputed based on localMinimumHeight
  minimumHeight?: number | null;

  // Sets the preferred height of the Node in the local coordinate frame.
  // NOTE: changing this or preferredHeight will adjust the other.
  // NOTE: when the Node's transform is updated, this value is recomputed based on preferredHeight
  localPreferredHeight?: number | null;

  // Sets the minimum height of the Node in the local coordinate frame. Usually set by the resizable Node itself to
  // indicate what preferred sizes are possible.
  // NOTE: changing this or minimumHeight will adjust the other.
  localMinimumHeight?: number | null;

  // Whether this component will have its preferred size set by things like layout containers.
  heightSizable?: boolean;
};

const HeightSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const clazz = class extends type {

    // parent/local preferred/minimum Properties. See the options above for more documentation
    readonly preferredHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly minimumHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly localPreferredHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly localMinimumHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly isHeightResizableProperty: TinyProperty<boolean> = new TinyProperty<boolean>( true );

    // Flags so that we can change one (parent/local) value and not enter an infinite loop changing the other
    _preferredHeightChanging = false;
    _minimumHeightChanging = false;

    constructor( ...args: any[] ) {
      super( ...args );

      const updatePreferred = this._updatePreferredHeight.bind( this );
      const updateLocalPreferred = this._updateLocalPreferredHeight.bind( this );
      const updateMinimum = this._updateMinimumHeight.bind( this );
      const updateLocalMinimum = this._updateLocalMinimumHeight.bind( this );

      // Update the opposite of parent/local when one changes
      this.preferredHeightProperty.lazyLink( updateLocalPreferred );
      this.localPreferredHeightProperty.lazyLink( updatePreferred );
      this.minimumHeightProperty.lazyLink( updateLocalMinimum );
      this.localMinimumHeightProperty.lazyLink( updateMinimum );

      // On a transform change, keep our local minimum (presumably unchanged), and our parent preferred size
      ( this as unknown as Node ).transformEmitter.addListener( updateLocalPreferred );
      ( this as unknown as Node ).transformEmitter.addListener( updateMinimum );
    }

    get preferredHeight(): number | null {
      return this.preferredHeightProperty.value;
    }

    set preferredHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredHeight should be null or a non-negative finite number' );

      this.preferredHeightProperty.value = value;
    }

    get localPreferredHeight(): number | null {
      return this.localPreferredHeightProperty.value;
    }

    set localPreferredHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'localPreferredHeight should be null or a non-negative finite number' );

      this.localPreferredHeightProperty.value = value;
    }

    get minimumHeight(): number | null {
      return this.minimumHeightProperty.value;
    }

    set minimumHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumHeightProperty.value = value;
    }

    get localMinimumHeight(): number | null {
      return this.localMinimumHeightProperty.value;
    }

    set localMinimumHeight( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.localMinimumHeightProperty.value = value;
    }

    get heightSizable(): boolean {
      return this.isHeightResizableProperty.value;
    }

    set heightSizable( value: boolean ) {
      this.isHeightResizableProperty.value = value;
    }

    get mixesHeightSizable(): boolean { return true; }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updateLocalPreferredHeight(): void {
      const node = this as unknown as Node;

      if ( !this._preferredHeightChanging ) {
        this._preferredHeightChanging = true;
        this.localPreferredHeightProperty.value = ( node.matrix.isAligned() && this.preferredHeight !== null )
                                                 ? Math.abs( node.transform.inverseDeltaY( this.preferredHeight ) )
                                                 : null;
        this._preferredHeightChanging = false;
      }
    }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updatePreferredHeight(): void {
      const node = this as unknown as Node;

      if ( !this._preferredHeightChanging ) {
        this._preferredHeightChanging = true;
        this.preferredHeightProperty.value = ( node.matrix.isAligned() && this.localPreferredHeight !== null )
                                            ? Math.abs( node.transform.transformDeltaY( this.localPreferredHeight ) )
                                            : null;
        this._preferredHeightChanging = false;
      }
    }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updateLocalMinimumHeight(): void {
      const node = this as unknown as Node;

      if ( !this._minimumHeightChanging ) {
        this._minimumHeightChanging = true;
        this.localMinimumHeightProperty.value = ( node.matrix.isAligned() && this.minimumHeight !== null )
                                               ? Math.abs( node.transform.inverseDeltaY( this.minimumHeight ) )
                                               : null;
        this._minimumHeightChanging = false;
      }
    }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updateMinimumHeight(): void {
      const node = this as unknown as Node;

      if ( !this._minimumHeightChanging ) {
        this._minimumHeightChanging = true;
        this.minimumHeightProperty.value = ( node.matrix.isAligned() && this.localMinimumHeight !== null )
                                          ? Math.abs( node.transform.transformDeltaY( this.localMinimumHeight ) )
                                          : null;
        this._minimumHeightChanging = false;
      }
    }
  };

  // If we're extending into a Node type, include option keys
  if ( type.prototype._mutatorKeys ) {
    const existingKeys = type.prototype._mutatorKeys;
    const newKeys = HEIGHT_SIZABLE_OPTION_KEYS;
    const indexOfBoundsBasedOptions = existingKeys.indexOf( REQUIRES_BOUNDS_OPTION_KEYS[ 0 ] );
    clazz.prototype._mutatorKeys = [
      ...existingKeys.slice( 0, indexOfBoundsBasedOptions ),
      ...newKeys,
      ...existingKeys.slice( indexOfBoundsBasedOptions )
    ];
  }

  return clazz;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as HeightSizable
const wrapper = () => HeightSizable( Node );
export type HeightSizableNode = InstanceType<ReturnType<typeof wrapper>>;
const isHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.heightSizable;
};
const mixesHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.mixesHeightSizable;
};

scenery.register( 'HeightSizable', HeightSizable );
export default HeightSizable;
export { isHeightSizable, mixesHeightSizable };
