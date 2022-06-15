// Copyright 2022, University of Colorado Boulder

/**
 * Provides a minimum and preferred width/height (both WidthSizable and HeightSizable, but with added features that
 * allow convenience of working with both dimensions at once).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import memoize from '../../../phet-core/js/memoize.js';
import { HEIGHT_SIZABLE_OPTION_KEYS, HeightSizable, HeightSizableOptions, Node, NodeOptions, REQUIRES_BOUNDS_OPTION_KEYS, scenery, WIDTH_SIZABLE_OPTION_KEYS, WidthSizable, WidthSizableOptions, DelayedMutate } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import Dimension2 from '../../../dot/js/Dimension2.js';
import assertMutuallyExclusiveOptions from '../../../phet-core/js/assertMutuallyExclusiveOptions.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';

export const SIZABLE_SELF_OPTION_KEYS = [
  'preferredSize',
  'minimumSize',
  'localPreferredSize',
  'localMinimumSize',
  'sizable'
];

export const SIZABLE_OPTION_KEYS = [
  'preferredSize',
  'minimumSize',
  'localPreferredSize',
  'localMinimumSize',
  'sizable',
  ...WIDTH_SIZABLE_OPTION_KEYS,
  ...HEIGHT_SIZABLE_OPTION_KEYS
];

type SelfOptions = {

  // Sets the preferred size of the Node in the parent coordinate frame. Nodes that implement this will attempt to keep
  // their `node.size` at this value. If null, the node will likely set its configuration to the minimum size.
  // NOTE: changing this or localPreferredHeight will adjust the other.
  // NOTE: preferredSize is not guaranteed currently. The component may end up having a smaller or larger size
  preferredSize?: Dimension2 | null;

  // Sets the minimum size of the Node in the parent coordinate frame. Usually not directly set by a client.
  // Usually a resizable Node will set its localMinimumSize (and that will get transferred to this value in the
  // parent coordinate frame).
  // NOTE: changing this or localMinimumSize will adjust the other.
  // NOTE: when the Node's transform is updated, this value is recomputed based on localMinimumSize
  minimumSize?: Dimension2 | null;

  // Sets the preferred size of the Node in the local coordinate frame.
  // NOTE: changing this or preferredSize will adjust the other.
  // NOTE: when the Node's transform is updated, this value is recomputed based on preferredSize
  localPreferredSize?: Dimension2 | null;

  // Sets the minimum size of the Node in the local coordinate frame. Usually set by the resizable Node itself to
  // indicate what preferred sizes are possible.
  // NOTE: changing this or minimumSize will adjust the other.
  localMinimumSize?: Dimension2 | null;

  // Whether this component will have its preferred size set by things like layout containers. If this is set to false,
  // it's recommended to set some sort of preferred size (so that it won't go to 0)
  sizable?: boolean;
};
type ParentOptions = WidthSizableOptions & HeightSizableOptions;
export type SizableOptions = SelfOptions & ParentOptions;

const Sizable = memoize( <SuperType extends Constructor>( type: SuperType ) => {
  const SuperExtendedType = WidthSizable( HeightSizable( type ) );
  const SizableMixin = DelayedMutate( 'Sizable', SIZABLE_SELF_OPTION_KEYS, class SizableMixin extends SuperExtendedType {

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );
    }

    public get preferredSize(): Dimension2 | null {
      assert && assert( ( this.preferredWidth === null ) === ( this.preferredHeight === null ),
        'Cannot get a preferredSize when one of preferredWidth/preferredHeight is null' );

      if ( this.preferredWidth === null || this.preferredHeight === null ) {
        return null;
      }
      else {
        return new Dimension2( this.preferredWidth, this.preferredHeight );
      }
    }

    public set preferredSize( value: Dimension2 | null ) {
      this.preferredWidth = value === null ? null : value.width;
      this.preferredHeight = value === null ? null : value.height;
    }

    public get localPreferredSize(): Dimension2 | null {
      assert && assert( ( this.localPreferredWidth === null ) === ( this.localPreferredHeight === null ),
        'Cannot get a preferredSize when one of preferredWidth/preferredHeight is null' );

      if ( this.localPreferredWidth === null || this.localPreferredHeight === null ) {
        return null;
      }
      else {
        return new Dimension2( this.localPreferredWidth, this.localPreferredHeight );
      }
    }

    public set localPreferredSize( value: Dimension2 | null ) {
      this.localPreferredWidth = value === null ? null : value.width;
      this.localPreferredHeight = value === null ? null : value.height;
    }

    public get minimumSize(): Dimension2 | null {
      assert && assert( ( this.minimumWidth === null ) === ( this.minimumHeight === null ),
        'Cannot get a minimumSize when one of minimumWidth/minimumHeight is null' );

      if ( this.minimumWidth === null || this.minimumHeight === null ) {
        return null;
      }
      else {
        return new Dimension2( this.minimumWidth, this.minimumHeight );
      }
    }

    public set minimumSize( value: Dimension2 | null ) {
      this.minimumWidth = value === null ? null : value.width;
      this.minimumHeight = value === null ? null : value.height;
    }

    public get localMinimumSize(): Dimension2 | null {
      assert && assert( ( this.localMinimumWidth === null ) === ( this.localMinimumHeight === null ),
        'Cannot get a minimumSize when one of minimumWidth/minimumHeight is null' );

      if ( this.localMinimumWidth === null || this.localMinimumHeight === null ) {
        return null;
      }
      else {
        return new Dimension2( this.localMinimumWidth, this.localMinimumHeight );
      }
    }

    public set localMinimumSize( value: Dimension2 | null ) {
      this.localMinimumWidth = value === null ? null : value.width;
      this.localMinimumHeight = value === null ? null : value.height;
    }

    public get sizable(): boolean {
      assert && assert( this.widthSizable === this.heightSizable,
        'widthSizable and heightSizable not the same, which is required for the sizable getter' );
      return this.widthSizable;
    }

    public set sizable( value: boolean ) {
      this.widthSizable = value;
      this.heightSizable = value;
    }

    public get mixesSizable(): boolean { return true; }

    validateLocalPreferredSize(): void {
      if ( assert ) {
        this.validateLocalPreferredWidth();
        this.validateLocalPreferredHeight();
      }
    }

    override mutate( options?: NodeOptions ): this {

      assertMutuallyExclusiveOptions( options, [ 'preferredSize' ], [ 'preferredWidth', 'preferredHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'localPreferredSize' ], [ 'localPreferredWidth', 'localPreferredHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'minimumSize' ], [ 'minimumWidth', 'minimumHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'localMinimumSize' ], [ 'localMinimumWidth', 'localMinimumHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'sizable' ], [ 'widthSizable', 'heightSizable' ] );

      // @ts-ignore
      return super.mutate( options );
    }
  } );

  // If we're extending into a Node type, include option keys
  if ( SuperExtendedType.prototype._mutatorKeys ) {
    const existingKeys = SuperExtendedType.prototype._mutatorKeys;
    const newKeys = SIZABLE_SELF_OPTION_KEYS;
    const indexOfBoundsBasedOptions = existingKeys.indexOf( REQUIRES_BOUNDS_OPTION_KEYS[ 0 ] );
    SizableMixin.prototype._mutatorKeys = [
      ...existingKeys.slice( 0, indexOfBoundsBasedOptions ),
      ...newKeys,
      ...existingKeys.slice( indexOfBoundsBasedOptions )
    ];
  }

  return SizableMixin;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as Sizable
// We need to define an unused function with a concrete type, so that we can extract the return type of the function
// and provide a type for a Node that mixes this type.
const wrapper = () => Sizable( Node );
export type SizableNode = InstanceType<ReturnType<typeof wrapper>>;

const isSizable = ( node: Node ): node is SizableNode => {
  return node.widthSizable && node.heightSizable;
};
const mixesSizable = ( node: Node ): node is SizableNode => {
  return node.mixesSizable;
};

scenery.register( 'Sizable', Sizable );
export default Sizable;
export { isSizable, mixesSizable };
