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
import { DelayedMutate, Node, REQUIRES_BOUNDS_OPTION_KEYS, scenery } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';

// Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

export const WIDTH_SIZABLE_OPTION_KEYS = [
  'preferredWidth',
  'minimumWidth',
  'localPreferredWidth',
  'localMinimumWidth',
  'widthSizable'
];

export type WidthSizableOptions = {
  // Sets the preferred width of the Node in the parent coordinate frame. Nodes that implement this will attempt to keep
  // their `node.width` at this value. If null, the node will likely set its configuration to the minimum width.
  // NOTE: changing this or localPreferredWidth will adjust the other.
  // NOTE: preferredWidth is not guaranteed currently. The component may end up having a smaller or larger size
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
  // NOTE: localPreferredWidth is not guaranteed currently. The component may end up having a smaller or larger size
  localPreferredWidth?: number | null;

  // Sets the minimum width of the Node in the local coordinate frame. Usually set by the resizable Node itself to
  // indicate what preferred sizes are possible.
  // NOTE: changing this or minimumWidth will adjust the other.
  localMinimumWidth?: number | null;

  // Whether this component will have its preferred size set by things like layout containers. If this is set to false,
  // it's recommended to set some sort of preferred size (so that it won't go to 0)
  widthSizable?: boolean;
};

// IMPORTANT: If you're combining this in, typically don't pass options that WidthSizable would take through the
// constructor. It will hit Node's mutate() likely, and then will fail because we haven't been able to set the
// values yet. If you're making something WidthSizable, please use a later mutate() to pass these options through.
// They WILL be caught by assertions if someone adds one of those options, but it could be a silent bug if no one
// is yet passing those options through.
const WidthSizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const WidthSizableTrait = DelayedMutate( 'WidthSizable', WIDTH_SIZABLE_OPTION_KEYS, class WidthSizableTrait extends type {

    // parent/local preferred/minimum Properties. See the options above for more documentation
    public readonly preferredWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly minimumWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly localPreferredWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly localMinimumWidthProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly isWidthResizableProperty: TinyProperty<boolean> = new TinyProperty<boolean>( true );

    // Flags so that we can change one (parent/local) value and not enter an infinite loop changing the others.
    // We want to lock out all other local or non-local preferred minimum sizes, whether in HeightSizable or WidthSizable
    // NOTE: We are merging declarations between HeightSizable and WidthSizable. If Sizable is used these flags
    // will be shared by both HeightSizable and WidthSizable.
    protected _preferredSizeChanging = false;
    protected _minimumSizeChanging = false;

    // Expose listeners, so that we'll be able to hook them up to the opposite dimension in Sizable
    protected _updatePreferredWidthListener: () => void;
    protected _updateLocalPreferredWidthListener: () => void;
    protected _updateMinimumWidthListener: () => void;
    protected _updateLocalMinimumWidthListener: () => void;

    // IMPORTANT: If you're combining this in, typically don't pass options that WidthSizable would take through the
    // constructor. It will hit Node's mutate() likely, and then will fail because we haven't been able to set the
    // values yet. If you're making something WidthSizable, please use a later mutate() to pass these options through.
    // They WILL be caught by assertions if someone adds one of those options, but it could be a silent bug if no one
    // is yet passing those options through.
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      this._updatePreferredWidthListener = this._updatePreferredWidth.bind( this );
      this._updateLocalPreferredWidthListener = this._updateLocalPreferredWidth.bind( this );
      this._updateMinimumWidthListener = this._updateMinimumWidth.bind( this );
      this._updateLocalMinimumWidthListener = this._updateLocalMinimumWidth.bind( this );

      // Update the opposite of parent/local when one changes
      this.preferredWidthProperty.lazyLink( this._updateLocalPreferredWidthListener );
      this.localPreferredWidthProperty.lazyLink( this._updatePreferredWidthListener );
      this.minimumWidthProperty.lazyLink( this._updateLocalMinimumWidthListener );
      this.localMinimumWidthProperty.lazyLink( this._updateMinimumWidthListener );

      // On a transform change, keep our local minimum (presumably unchanged), and our parent preferred size
      ( this as unknown as Node ).transformEmitter.addListener( this._updateLocalPreferredWidthListener );
      // On a transform change this should update the minimum
      ( this as unknown as Node ).transformEmitter.addListener( this._updateMinimumWidthListener );
    }

    public get preferredWidth(): number | null {
      assert && assert( this.preferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );

      return this.preferredWidthProperty.value;
    }

    public set preferredWidth( value: number | null ) {
      assert && assert( this.preferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredWidth should be null or a non-negative finite number' );

      this.preferredWidthProperty.value = value;
    }

    public get localPreferredWidth(): number | null {
      assert && assert( this.localPreferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.localPreferredWidthProperty.value;
    }

    public set localPreferredWidth( value: number | null ) {
      assert && assert( this.localPreferredWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'localPreferredWidth should be null or a non-negative finite number' );

      this.localPreferredWidthProperty.value = value;
    }

    public get minimumWidth(): number | null {
      assert && assert( this.minimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.minimumWidthProperty.value;
    }

    public set minimumWidth( value: number | null ) {
      assert && assert( this.minimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumWidthProperty.value = value;
    }

    public get localMinimumWidth(): number | null {
      assert && assert( this.localMinimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.localMinimumWidthProperty.value;
    }

    public set localMinimumWidth( value: number | null ) {
      assert && assert( this.localMinimumWidthProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.localMinimumWidthProperty.value = value;
    }

    public get widthSizable(): boolean {
      assert && assert( this.isWidthResizableProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.isWidthResizableProperty.value;
    }

    public set widthSizable( value: boolean ) {
      assert && assert( this.isWidthResizableProperty,
        'WidthSizable options should be set from a later mutate() call instead of the super constructor' );
      this.isWidthResizableProperty.value = value;
    }

    public get extendsWidthSizable(): boolean { return true; }

    public validateLocalPreferredWidth(): void {
      if ( assert ) {
        const currentWidth = ( this as unknown as Node ).localWidth;
        const effectiveMinimumWidth = this.localMinimumWidth === null ? currentWidth : this.localMinimumWidth;
        const idealWidth = this.localPreferredWidth === null ? effectiveMinimumWidth : this.localPreferredWidth;

        // Handle non-finite values with exact equality
        assert( idealWidth === currentWidth || Math.abs( idealWidth - currentWidth ) < 1e-7 );
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculateLocalPreferredWidth(): number | null {
      const node = this as unknown as Node;

      return ( node.matrix.isAligned() && this.preferredWidth !== null )
             ? Math.abs( node.transform.inverseDeltaX( this.preferredWidth ) )
             : null;
    }

    private _updateLocalPreferredWidth(): void {
      assert && ( this as unknown as Node ).auditMaxDimensions();

      if ( !this._preferredSizeChanging ) {
        this._preferredSizeChanging = true;

        const localPreferredWidth = this._calculateLocalPreferredWidth();

        if ( this.localPreferredWidthProperty.value === null ||
             localPreferredWidth === null ||
             Math.abs( this.localPreferredWidthProperty.value - localPreferredWidth ) > CHANGE_POSITION_THRESHOLD ) {
          this.localPreferredWidthProperty.value = localPreferredWidth;
        }
        this._preferredSizeChanging = false;
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculatePreferredWidth(): number | null {
      const node = this as unknown as Node;

      return ( node.matrix.isAligned() && this.localPreferredWidth !== null )
             ? Math.abs( node.transform.transformDeltaX( this.localPreferredWidth ) )
             : null;
    }

    private _updatePreferredWidth(): void {
      if ( !this._preferredSizeChanging ) {
        this._preferredSizeChanging = true;

        const preferredWidth = this._calculatePreferredWidth();

        if ( this.preferredWidthProperty.value === null ||
             preferredWidth === null ||
             Math.abs( this.preferredWidthProperty.value - preferredWidth ) > CHANGE_POSITION_THRESHOLD ) {
          this.preferredWidthProperty.value = preferredWidth;
        }
        this._preferredSizeChanging = false;
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculateLocalMinimumWidth(): number | null {
      const node = this as unknown as Node;

      return ( node.matrix.isAligned() && this.minimumWidth !== null )
             ? Math.abs( node.transform.inverseDeltaX( this.minimumWidth ) )
             : null;
    }

    private _updateLocalMinimumWidth(): void {
      if ( !this._minimumSizeChanging ) {
        this._minimumSizeChanging = true;

        const localMinimumWidth = this._calculateLocalMinimumWidth();

        if ( this.localMinimumWidthProperty.value === null ||
             localMinimumWidth === null ||
             Math.abs( this.localMinimumWidthProperty.value - localMinimumWidth ) > CHANGE_POSITION_THRESHOLD ) {
          this.localMinimumWidthProperty.value = localMinimumWidth;
        }
        this._minimumSizeChanging = false;
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculateMinimumWidth(): number | null {
      const node = this as unknown as Node;

      return ( node.matrix.isAligned() && this.localMinimumWidth !== null )
             ? Math.abs( node.transform.transformDeltaX( this.localMinimumWidth ) )
             : null;
    }

    private _updateMinimumWidth(): void {
      if ( !this._minimumSizeChanging ) {
        this._minimumSizeChanging = true;

        const minimumWidth = this._calculateMinimumWidth();

        if ( this.minimumWidthProperty.value === null ||
             minimumWidth === null ||
             Math.abs( this.minimumWidthProperty.value - minimumWidth ) > CHANGE_POSITION_THRESHOLD ) {
          this.minimumWidthProperty.value = minimumWidth;
        }
        this._minimumSizeChanging = false;
      }
    }
  } );

  // If we're extending into a Node type, include option keys
  if ( type.prototype._mutatorKeys ) {
    const existingKeys = type.prototype._mutatorKeys;
    const newKeys = WIDTH_SIZABLE_OPTION_KEYS;
    const indexOfBoundsBasedOptions = existingKeys.indexOf( REQUIRES_BOUNDS_OPTION_KEYS[ 0 ] );
    WidthSizableTrait.prototype._mutatorKeys = [
      ...existingKeys.slice( 0, indexOfBoundsBasedOptions ),
      ...newKeys,
      ...existingKeys.slice( indexOfBoundsBasedOptions )
    ];
  }

  return WidthSizableTrait;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as widthSizable
// We need to define an unused function with a concrete type, so that we can extract the return type of the function
// and provide a type for a Node that extends this type.
const wrapper = () => WidthSizable( Node );
export type WidthSizableNode = InstanceType<ReturnType<typeof wrapper>>;

const isWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.widthSizable;
};
const extendsWidthSizable = ( node: Node ): node is WidthSizableNode => {
  return node.extendsWidthSizable;
};

scenery.register( 'WidthSizable', WidthSizable );
export default WidthSizable;
export { isWidthSizable, extendsWidthSizable };
