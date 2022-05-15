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
import { scenery, Node, REQUIRES_BOUNDS_OPTION_KEYS } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';

export const WIDTH_SIZABLE_OPTION_KEYS = [
  'preferredWidth',
  'minimumWidth',
  'localPreferredWidth',
  'localMinimumWidth',
  'widthSizable'
];

export type WidthSizableSelfOptions = {
  // Sets the preferred width of the Node in the parent coordinate frame. Nodes that implement this will attempt to keep
  // their `node.width` at this value. If null, the node will likely set its configuration to the minimum width.
  // NOTE: changing this or localPreferredWidth will adjust the other.
  // NOTE: preferredHeight is not guaranteed currently. The component may end up having a smaller or larger size
  preferredWidth?: number | null;

  // Sets the minimum width of the Node in the parent coordinate frame. Usually not directly set by a client.
  // Usually a resizable Node will set its localMinimumWidth (and that will get transferred to this value in the
  // parent coordinate frame).
  // NOTE: changing this or localMinimumWidth will adjust the other.
  // NOTE: when the Node's transform is updated, this value is recomputed based on localMinimumWidth
  minimumWidth?: number | null;

  // Sets the preferred width of the Node in the local coordinate frame.
  // NOTE: changing this or preferredWidth will adjust the other.
  // NOTE: when the Node's transform is updated, this value is recomputed based on preferredWidth
  localPreferredWidth?: number | null;

  // Sets the minimum width of the Node in the local coordinate frame. Usually set by the resizable Node itself to
  // indicate what preferred sizes are possible.
  // NOTE: changing this or minimumWidth will adjust the other.
  localMinimumWidth?: number | null;

  // Whether this component will have its preferred size set by things like layout containers. If this is set to false,
  // it's recommended to set some sort of preferred size (so that it won't go to 0)
  widthSizable?: boolean;
};

// IMPORTANT: If you're mixing this in, typically don't pass options that WidthSizable would take through the
// constructor. It will hit Node's mutate() likely, and then will fail because we haven't been able to set the
// values yet. If you're making something WidthSizable, please use a later mutate() to pass these options through.
// They WILL be caught by assertions if someone adds one of those options, but it could be a silent bug if no one
// is yet passing those options through.
const WidthSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const clazz = class extends type {

    // parent/local preferred/minimum Properties. See the options above for more documentation
    readonly preferredWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly minimumWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly localPreferredWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly localMinimumWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly isWidthResizableProperty: TinyProperty<boolean> = new TinyProperty<boolean>( true );

    // Flags so that we can change one (parent/local) value and not enter an infinite loop changing the other
    _preferredWidthChanging = false;
    _minimumWidthChanging = false;

    // IMPORTANT: If you're mixing this in, typically don't pass options that WidthSizable would take through the
    // constructor. It will hit Node's mutate() likely, and then will fail because we haven't been able to set the
    // values yet. If you're making something WidthSizable, please use a later mutate() to pass these options through.
    // They WILL be caught by assertions if someone adds one of those options, but it could be a silent bug if no one
    // is yet passing those options through.
    constructor( ...args: any[] ) {
      super( ...args );

      const updatePreferred = this._updatePreferredWidth.bind( this );
      const updateLocalPreferred = this._updateLocalPreferredWidth.bind( this );
      const updateMinimum = this._updateMinimumWidth.bind( this );
      const updateLocalMinimum = this._updateLocalMinimumWidth.bind( this );

      // Update the opposite of parent/local when one changes
      this.preferredWidthProperty.lazyLink( updateLocalPreferred );
      this.localPreferredWidthProperty.lazyLink( updatePreferred );
      this.minimumWidthProperty.lazyLink( updateLocalMinimum );
      this.localMinimumWidthProperty.lazyLink( updateMinimum );

      // On a transform change, keep our local minimum (presumably unchanged), and our parent preferred size
      ( this as unknown as Node ).transformEmitter.addListener( updateLocalPreferred );
      ( this as unknown as Node ).transformEmitter.addListener( updateMinimum );
    }

    get preferredWidth(): number | null {
      assert && assert( this.preferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );

      return this.preferredWidthProperty.value;
    }

    set preferredWidth( value: number | null ) {
      assert && assert( this.preferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredWidth should be null or a non-negative finite number' );

      this.preferredWidthProperty.value = value;
    }

    get localPreferredWidth(): number | null {
      assert && assert( this.localPreferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.localPreferredWidthProperty.value;
    }

    set localPreferredWidth( value: number | null ) {
      assert && assert( this.localPreferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'localPreferredWidth should be null or a non-negative finite number' );

      this.localPreferredWidthProperty.value = value;
    }

    get minimumWidth(): number | null {
      assert && assert( this.minimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.minimumWidthProperty.value;
    }

    set minimumWidth( value: number | null ) {
      assert && assert( this.minimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumWidthProperty.value = value;
    }

    get localMinimumWidth(): number | null {
      assert && assert( this.localMinimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.localMinimumWidthProperty.value;
    }

    set localMinimumWidth( value: number | null ) {
      assert && assert( this.localMinimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.localMinimumWidthProperty.value = value;
    }

    get widthSizable(): boolean {
      assert && assert( this.isWidthResizableProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.isWidthResizableProperty.value;
    }

    set widthSizable( value: boolean ) {
      assert && assert( this.isWidthResizableProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      this.isWidthResizableProperty.value = value;
    }

    get mixesWidthSizable(): boolean { return true; }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updateLocalPreferredWidth(): void {
      const node = this as unknown as Node;

      if ( !this._preferredWidthChanging ) {
        this._preferredWidthChanging = true;
        this.localPreferredWidthProperty.value = ( node.matrix.isAligned() && this.preferredWidth !== null )
                                                 ? Math.abs( node.transform.inverseDeltaX( this.preferredWidth ) )
                                                 : null;
        this._preferredWidthChanging = false;
      }
    }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updatePreferredWidth(): void {
      const node = this as unknown as Node;

      if ( !this._preferredWidthChanging ) {
        this._preferredWidthChanging = true;
        this.preferredWidthProperty.value = ( node.matrix.isAligned() && this.localPreferredWidth !== null )
                                            ? Math.abs( node.transform.transformDeltaX( this.localPreferredWidth ) )
                                            : null;
        this._preferredWidthChanging = false;
      }
    }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updateLocalMinimumWidth(): void {
      const node = this as unknown as Node;

      if ( !this._minimumWidthChanging ) {
        this._minimumWidthChanging = true;
        this.localMinimumWidthProperty.value = ( node.matrix.isAligned() && this.minimumWidth !== null )
                                               ? Math.abs( node.transform.inverseDeltaX( this.minimumWidth ) )
                                               : null;
        this._minimumWidthChanging = false;
      }
    }

    // Used internally, do not call (can't be private due to TypeScript mixin constraints)
    _updateMinimumWidth(): void {
      const node = this as unknown as Node;

      if ( !this._minimumWidthChanging ) {
        this._minimumWidthChanging = true;
        this.minimumWidthProperty.value = ( node.matrix.isAligned() && this.localMinimumWidth !== null )
                                          ? Math.abs( node.transform.transformDeltaX( this.localMinimumWidth ) )
                                          : null;
        this._minimumWidthChanging = false;
      }
    }
  };

  // If we're extending into a Node type, include option keys
  if ( type.prototype._mutatorKeys ) {
    const existingKeys = type.prototype._mutatorKeys;
    const newKeys = WIDTH_SIZABLE_OPTION_KEYS;
    const indexOfBoundsBasedOptions = existingKeys.indexOf( REQUIRES_BOUNDS_OPTION_KEYS[ 0 ] );
    clazz.prototype._mutatorKeys = [
      ...existingKeys.slice( 0, indexOfBoundsBasedOptions ),
      ...newKeys,
      ...existingKeys.slice( indexOfBoundsBasedOptions )
    ];
  }

  return clazz;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as widthSizable
const wrapper = () => WidthSizable( Node );
export type WidthSizableNode = InstanceType<ReturnType<typeof wrapper>>;
const isWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.widthSizable;
};
const mixesWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.mixesWidthSizable;
};

scenery.register( 'WidthSizable', WidthSizable );
export default WidthSizable;
export { isWidthSizable, mixesWidthSizable };
