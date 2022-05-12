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

export const WIDTH_SIZABLE_OPTION_KEYS = [
  'preferredWidth',
  'minimumWidth',
  'localPreferredWidth',
  'localMinimumWidth'
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
};

const WidthSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const clazz = class extends type {

    // parent/local preferred/minimum Properties. See the options above for more documentation
    readonly preferredWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly minimumWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly localPreferredWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    readonly localMinimumWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );

    // Flags so that we can change one (parent/local) value and not enter an infinite loop changing the other
    _preferredWidthChanging = false;
    _minimumWidthChanging = false;

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
      return this.preferredWidthProperty.value;
    }

    set preferredWidth( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredWidth should be null or a non-negative finite number' );

      this.preferredWidthProperty.value = value;
    }

    get localPreferredWidth(): number | null {
      return this.localPreferredWidthProperty.value;
    }

    set localPreferredWidth( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'localPreferredWidth should be null or a non-negative finite number' );

      this.localPreferredWidthProperty.value = value;
    }

    get minimumWidth(): number | null {
      return this.minimumWidthProperty.value;
    }

    set minimumWidth( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumWidthProperty.value = value;
    }

    get localMinimumWidth(): number | null {
      return this.localMinimumWidthProperty.value;
    }

    set localMinimumWidth( value: number | null ) {
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.localMinimumWidthProperty.value = value;
    }

    // Detection flag for this trait
    get widthSizable(): boolean { return true; }

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
    clazz.prototype._mutatorKeys = type.prototype._mutatorKeys.concat( WIDTH_SIZABLE_OPTION_KEYS );
  }

  return clazz;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as widthSizable
const wrapper = () => WidthSizable( Node );
export type WidthSizableNode = InstanceType<ReturnType<typeof wrapper>>;
const isWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.widthSizable;
};

scenery.register( 'WidthSizable', WidthSizable );
export default WidthSizable;
export { isWidthSizable };
