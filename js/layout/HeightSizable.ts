// Copyright 2021-2023, University of Colorado Boulder

/**
 * HeightSizable is a trait that provides a minimum and preferred height. The minimum height is set by the component,
 * so that layout containers could know how "small" the component can be made. The preferred height is set by the
 * layout container, and the component should adjust its size so that it takes up that height.
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

export const HEIGHT_SIZABLE_OPTION_KEYS = [
  'preferredHeight',
  'minimumHeight',
  'localPreferredHeight',
  'localMinimumHeight',
  'heightSizable'
];

export type HeightSizableOptions = {
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
  // NOTE: localPreferredHeight is not guaranteed currently. The component may end up having a smaller or larger size
  localPreferredHeight?: number | null;

  // Sets the minimum height of the Node in the local coordinate frame. Usually set by the resizable Node itself to
  // indicate what preferred sizes are possible.
  // NOTE: changing this or minimumHeight will adjust the other.
  localMinimumHeight?: number | null;

  // Whether this component will have its preferred size set by things like layout containers. If this is set to false,
  // it's recommended to set some sort of preferred size (so that it won't go to 0)
  heightSizable?: boolean;
};

// IMPORTANT: If you're combining this in, typically don't pass options that HeightSizable would take through the
// constructor. It will hit Node's mutate() likely, and then will fail because we haven't been able to set the
// values yet. If you're making something HeightSizable, please use a later mutate() to pass these options through.
// They WILL be caught by assertions if someone adds one of those options, but it could be a silent bug if no one
// is yet passing those options through.
const HeightSizable = memoize( <SuperType extends Constructor<Node>>( type: SuperType ) => {
  const HeightSizableTrait = DelayedMutate( 'HeightSizable', HEIGHT_SIZABLE_OPTION_KEYS, class HeightSizableTrait extends type {

    // parent/local preferred/minimum Properties. See the options above for more documentation
    public readonly preferredHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly minimumHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly localPreferredHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly localMinimumHeightProperty: TinyProperty<number | null> = new TinyProperty<number | null>( null );
    public readonly isHeightResizableProperty: TinyProperty<boolean> = new TinyProperty<boolean>( true );

    // Flags so that we can change one (parent/local) value and not enter an infinite loop changing the others.
    // We want to lock out all other local or non-local preferred minimum sizes, whether in HeightSizable or WidthSizable
    // NOTE: We are merging declarations between HeightSizable and WidthSizable. If Sizable is used these flags
    // will be shared by both HeightSizable and WidthSizable.
    protected _preferredSizeChanging = false;
    protected _minimumSizeChanging = false;

    // We'll need to detect reentrancy when setting the dual of the preferred/minimum properties (e.g. local vs parent).
    // If we get a reentrant case, we'll need to detect it and clear things up at the end (updating the minimum size
    // in the parent coordinate frame, and the preferred size in the local coordinate frame).
    // An example is if the minimum size is set, and that triggers a listener that UPDATES something that changes the
    // minimum size, we'll need to make sure that the local minimum size is updated AFTER everything has happened.
    // These locks are used to detect these cases, and then run the appropriate updates afterward to make sure that the
    // local and parent values are in sync (based on the transform used).
    protected _preferredSizeChangeAttemptDuringLock = false;
    protected _minimumSizeChangeAttemptDuringLock = false;

    // Expose listeners, so that we'll be able to hook them up to the opposite dimension in Sizable
    protected _updatePreferredHeightListener: () => void;
    protected _updateLocalPreferredHeightListener: () => void;
    protected _updateMinimumHeightListener: () => void;
    protected _updateLocalMinimumHeightListener: () => void;

    // IMPORTANT: If you're combining this in, typically don't pass options that HeightSizable would take through the
    // constructor. It will hit Node's mutate() likely, and then will fail because we haven't been able to set the
    // values yet. If you're making something HeightSizable, please use a later mutate() to pass these options through.
    // They WILL be caught by assertions if someone adds one of those options, but it could be a silent bug if no one
    // is yet passing those options through.
    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      this._updatePreferredHeightListener = this._updatePreferredHeight.bind( this );
      this._updateLocalPreferredHeightListener = this._updateLocalPreferredHeight.bind( this );
      this._updateMinimumHeightListener = this._updateMinimumHeight.bind( this );
      this._updateLocalMinimumHeightListener = this._updateLocalMinimumHeight.bind( this );

      // Update the opposite of parent/local when one changes
      this.preferredHeightProperty.lazyLink( this._updateLocalPreferredHeightListener );
      this.localPreferredHeightProperty.lazyLink( this._updatePreferredHeightListener );
      this.minimumHeightProperty.lazyLink( this._updateLocalMinimumHeightListener );
      this.localMinimumHeightProperty.lazyLink( this._updateMinimumHeightListener );

      // On a transform change, keep our local minimum (presumably unchanged), and our parent preferred size
      this.transformEmitter.addListener( this._updateLocalPreferredHeightListener );
      // On a transform change this should update the minimum
      this.transformEmitter.addListener( this._updateMinimumHeightListener );
    }

    public get preferredHeight(): number | null {
      assert && assert( this.preferredHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.preferredHeightProperty.value;
    }

    public set preferredHeight( value: number | null ) {
      assert && assert( this.preferredHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'preferredHeight should be null or a non-negative finite number' );

      this.preferredHeightProperty.value = value;
    }

    public get localPreferredHeight(): number | null {
      assert && assert( this.localPreferredHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.localPreferredHeightProperty.value;
    }

    public set localPreferredHeight( value: number | null ) {
      assert && assert( this.localPreferredHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) && value >= 0 ),
        'localPreferredHeight should be null or a non-negative finite number' );

      this.localPreferredHeightProperty.value = value;
    }

    public get minimumHeight(): number | null {
      assert && assert( this.minimumHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.minimumHeightProperty.value;
    }

    public set minimumHeight( value: number | null ) {
      assert && assert( this.minimumHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.minimumHeightProperty.value = value;
    }

    public get localMinimumHeight(): number | null {
      assert && assert( this.localMinimumHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.localMinimumHeightProperty.value;
    }

    public set localMinimumHeight( value: number | null ) {
      assert && assert( this.localMinimumHeightProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      assert && assert( value === null || ( typeof value === 'number' && isFinite( value ) ) );

      this.localMinimumHeightProperty.value = value;
    }

    public override get heightSizable(): boolean {
      assert && assert( this.isHeightResizableProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      return this.isHeightResizableProperty.value;
    }

    public override set heightSizable( value: boolean ) {
      assert && assert( this.isHeightResizableProperty,
        'HeightSizable options should be set from a later mutate() call instead of the super constructor' );
      this.isHeightResizableProperty.value = value;
    }

    public override get extendsHeightSizable(): boolean { return true; }

    public validateLocalPreferredHeight(): void {
      if ( assert ) {
        const currentHeight = this.localHeight;
        const effectiveMinimumHeight = this.localMinimumHeight === null ? currentHeight : this.localMinimumHeight;
        const idealHeight = this.localPreferredHeight === null ? effectiveMinimumHeight : this.localPreferredHeight;

        // Handle non-finite values with exact equality
        assert( idealHeight === currentHeight || Math.abs( idealHeight - currentHeight ) < 1e-7 );
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculateLocalPreferredHeight(): number | null {

      return ( this.matrix.isAligned() && this.preferredHeight !== null )
             ? Math.abs( this.transform.inverseDeltaY( this.preferredHeight ) )
             : null;
    }

    // Provides a hook to Sizable, since we'll need to cross-link this to also try updating the opposite dimension
    protected _onReentrantPreferredHeight(): void {
      this._updateLocalPreferredHeightListener();
    }

    private _updateLocalPreferredHeight(): void {
      assert && this.auditMaxDimensions();

      if ( !this._preferredSizeChanging ) {
        this._preferredSizeChanging = true;

        // Since the local "preferred" size is the one that we'll want to continue to update if we experience
        // reentrancy (since we treat the non-local version as the ground truth), we'll loop here until we didn't get
        // an attempt to change it. This will ensure that after changes, we'll have a consistent preferred and
        // localPreferred size.
        do {
          this._preferredSizeChangeAttemptDuringLock = false;

          const localPreferredHeight = this._calculateLocalPreferredHeight();

          if ( this.localPreferredHeightProperty.value === null ||
               localPreferredHeight === null ||
               Math.abs( this.localPreferredHeightProperty.value - localPreferredHeight ) > CHANGE_POSITION_THRESHOLD ) {
            this.localPreferredHeightProperty.value = localPreferredHeight;
          }
        }
        while ( this._preferredSizeChangeAttemptDuringLock );

        this._preferredSizeChanging = false;
      }
      else {
        this._preferredSizeChangeAttemptDuringLock = true;
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculatePreferredHeight(): number | null {

      return ( this.matrix.isAligned() && this.localPreferredHeight !== null )
             ? Math.abs( this.transform.transformDeltaY( this.localPreferredHeight ) )
             : null;
    }

    private _updatePreferredHeight(): void {
      if ( !this._preferredSizeChanging ) {
        this._preferredSizeChanging = true;

        this._preferredSizeChangeAttemptDuringLock = false;

        const preferredHeight = this._calculatePreferredHeight();

        if ( this.preferredHeightProperty.value === null ||
             preferredHeight === null ||
             Math.abs( this.preferredHeightProperty.value - preferredHeight ) > CHANGE_POSITION_THRESHOLD ) {
          this.preferredHeightProperty.value = preferredHeight;
        }
        this._preferredSizeChanging = false;

        // Here, in the case of reentrance, we'll actually want to switch to updating the local preferred size, since
        // given any other changes it should be the primary one to change.
        if ( this._preferredSizeChangeAttemptDuringLock ) {
          this._onReentrantPreferredHeight();
        }
      }
      else {
        this._preferredSizeChangeAttemptDuringLock = true;
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculateLocalMinimumHeight(): number | null {
      return ( this.matrix.isAligned() && this.minimumHeight !== null )
             ? Math.abs( this.transform.inverseDeltaY( this.minimumHeight ) )
             : null;
    }

    protected _onReentrantLocalMinimumHeight(): void {
      this._updateMinimumHeight();
    }

    private _updateLocalMinimumHeight(): void {
      if ( !this._minimumSizeChanging ) {
        this._minimumSizeChanging = true;

        this._minimumSizeChangeAttemptDuringLock = false;

        const localMinimumHeight = this._calculateLocalMinimumHeight();

        if ( this.localMinimumHeightProperty.value === null ||
             localMinimumHeight === null ||
             Math.abs( this.localMinimumHeightProperty.value - localMinimumHeight ) > CHANGE_POSITION_THRESHOLD ) {
          this.localMinimumHeightProperty.value = localMinimumHeight;
        }
        this._minimumSizeChanging = false;

        // Here, in the case of reentrance, we'll actually want to switch to updating the non-local minimum size, since
        // given any other changes it should be the primary one to change.
        if ( this._minimumSizeChangeAttemptDuringLock ) {
          this._onReentrantLocalMinimumHeight();
        }
      }
      else {
        this._minimumSizeChangeAttemptDuringLock = true;
      }
    }

    // This is provided to hook into the Sizable trait, so that we can update the opposite dimension
    protected _calculateMinimumHeight(): number | null {
      return ( this.matrix.isAligned() && this.localMinimumHeight !== null )
             ? Math.abs( this.transform.transformDeltaY( this.localMinimumHeight ) )
             : null;
    }

    private _updateMinimumHeight(): void {
      if ( !this._minimumSizeChanging ) {
        this._minimumSizeChanging = true;

        // Since the non-local "minimum" size is the one that we'll want to continue to update if we experience
        // reentrancy (since we treat the local version as the ground truth), we'll loop here until we didn't get
        // an attempt to change it. This will ensure that after changes, we'll have a consistent minimum and
        // localMinimum size.
        do {
          this._minimumSizeChangeAttemptDuringLock = false;

          const minimumHeight = this._calculateMinimumHeight();

          if ( this.minimumHeightProperty.value === null ||
               minimumHeight === null ||
               Math.abs( this.minimumHeightProperty.value - minimumHeight ) > CHANGE_POSITION_THRESHOLD ) {
            this.minimumHeightProperty.value = minimumHeight;
          }
        }
        while ( this._minimumSizeChangeAttemptDuringLock );

        this._minimumSizeChanging = false;
      }
    }

    public override mutate( options?: HeightSizableOptions & Parameters<InstanceType<SuperType>[ 'mutate' ]>[ 0 ] ): this {
      return super.mutate( options );
    }
  } );

  // If we're extending into a Node type, include option keys
  if ( type.prototype._mutatorKeys ) {
    const existingKeys = type.prototype._mutatorKeys;
    const newKeys = HEIGHT_SIZABLE_OPTION_KEYS;
    const indexOfBoundsBasedOptions = existingKeys.indexOf( REQUIRES_BOUNDS_OPTION_KEYS[ 0 ] );
    HeightSizableTrait.prototype._mutatorKeys = [
      ...existingKeys.slice( 0, indexOfBoundsBasedOptions ),
      ...newKeys,
      ...existingKeys.slice( indexOfBoundsBasedOptions )
    ];
  }

  return HeightSizableTrait;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as HeightSizable.
// We need to define an unused function with a concrete type, so that we can extract the return type of the function
// and provide a type for a Node that extends this type.
const wrapper = () => HeightSizable( Node );
export type HeightSizableNode = InstanceType<ReturnType<typeof wrapper>>;

const isHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.heightSizable;
};
const extendsHeightSizable = ( node: Node ): node is HeightSizableNode => {
  return node.extendsHeightSizable;
};

scenery.register( 'HeightSizable', HeightSizable );
export default HeightSizable;
export { isHeightSizable, extendsHeightSizable };
