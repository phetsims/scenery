// Copyright 2022-2023, University of Colorado Boulder

/**
 * Sizable is a trait that provides a minimum and preferred width/height (both WidthSizable and HeightSizable,
 * but with added features that allow convenience of working with both dimensions at once).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import memoize from '../../../phet-core/js/memoize.js';
import { DelayedMutate, HEIGHT_SIZABLE_OPTION_KEYS, HeightSizable, HeightSizableOptions, Node, REQUIRES_BOUNDS_OPTION_KEYS, scenery, WIDTH_SIZABLE_OPTION_KEYS, WidthSizable, WidthSizableOptions } from '../imports.js';
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

const Sizable = memoize( <SuperType extends Constructor<Node>>( type: SuperType ) => {
  const SuperExtendedType = WidthSizable( HeightSizable( type ) );
  const SizableTrait = DelayedMutate( 'Sizable', SIZABLE_SELF_OPTION_KEYS, class SizableTrait extends SuperExtendedType {

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      // We've added code to conditionally update the preferred/minimum opposite dimensions, so we'll need to
      // cross-link the listeners we've created in WidthSizable/HeightSizable

      this.preferredWidthProperty.lazyLink( this._updateLocalPreferredHeightListener );
      this.preferredHeightProperty.lazyLink( this._updateLocalPreferredWidthListener );

      this.localPreferredWidthProperty.lazyLink( this._updatePreferredHeightListener );
      this.localPreferredHeightProperty.lazyLink( this._updatePreferredWidthListener );

      this.minimumWidthProperty.lazyLink( this._updateLocalMinimumHeightListener );
      this.minimumHeightProperty.lazyLink( this._updateLocalMinimumWidthListener );

      this.localMinimumWidthProperty.lazyLink( this._updateMinimumHeightListener );
      this.localMinimumHeightProperty.lazyLink( this._updateMinimumWidthListener );
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

    public override get extendsSizable(): boolean { return true; }

    public validateLocalPreferredSize(): void {
      if ( assert ) {
        this.validateLocalPreferredWidth();
        this.validateLocalPreferredHeight();
      }
    }

    public override mutate( options?: SelfOptions & Parameters<InstanceType<SuperType>[ 'mutate' ]>[ 0 ] ): this {

      assertMutuallyExclusiveOptions( options, [ 'preferredSize' ], [ 'preferredWidth', 'preferredHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'localPreferredSize' ], [ 'localPreferredWidth', 'localPreferredHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'minimumSize' ], [ 'minimumWidth', 'minimumHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'localMinimumSize' ], [ 'localMinimumWidth', 'localMinimumHeight' ] );
      assertMutuallyExclusiveOptions( options, [ 'sizable' ], [ 'widthSizable', 'heightSizable' ] );

      return super.mutate( options );
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculateLocalPreferredWidth(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.preferredWidth !== null ) {
            return Math.abs( this.transform.inverseDeltaX( this.preferredWidth ) );
          }
        }
        // If we're height-sizable and we have an orientation swap, set the correct preferred width!
        else if ( this.preferredHeight !== null ) {
          return Math.abs( this.transform.getInverse().m01() * this.preferredHeight );
        }
      }

      return null;
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculateLocalPreferredHeight(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.preferredHeight !== null ) {
            return Math.abs( this.transform.inverseDeltaY( this.preferredHeight ) );
          }
        }
        // If we're width-sizable and we have an orientation swap, set the correct preferred height!
        else if ( this.preferredWidth !== null ) {
          return Math.abs( this.transform.getInverse().m10() * this.preferredWidth );
        }
      }

      return null;
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculatePreferredWidth(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.localPreferredWidth !== null ) {
            return Math.abs( this.transform.transformDeltaX( this.localPreferredWidth ) );
          }
        }
        else if ( this.localPreferredHeight !== null ) {
          return Math.abs( this.transform.matrix.m01() * this.localPreferredHeight );
        }
      }

      return null;
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculatePreferredHeight(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.localPreferredHeight !== null ) {
            return Math.abs( this.transform.transformDeltaY( this.localPreferredHeight ) );
          }
        }
        else if ( this.localPreferredWidth !== null ) {
          return Math.abs( this.transform.matrix.m10() * this.localPreferredWidth );
        }
      }

      return null;
    }

    // We'll need to cross-link because we might need to update either the width or height when the other changes
    protected override _onReentrantLocalMinimumWidth(): void {
      this._updateMinimumWidthListener();
      this._updateMinimumHeightListener();
    }

    // We'll need to cross-link because we might need to update either the width or height when the other changes
    protected override _onReentrantLocalMinimumHeight(): void {
      this._updateMinimumWidthListener();
      this._updateMinimumHeightListener();
    }

    // We'll need to cross-link because we might need to update either the width or height when the other changes
    protected override _onReentrantPreferredWidth(): void {
      this._updateLocalPreferredWidthListener();
      this._updateLocalPreferredHeightListener();
    }

    // We'll need to cross-link because we might need to update either the width or height when the other changes
    protected override _onReentrantPreferredHeight(): void {
      this._updateLocalPreferredWidthListener();
      this._updateLocalPreferredHeightListener();
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculateLocalMinimumWidth(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.minimumWidth !== null ) {
            return Math.abs( this.transform.inverseDeltaX( this.minimumWidth ) );
          }
        }
        else if ( this.minimumHeight !== null ) {
          return Math.abs( this.transform.getInverse().m01() * this.minimumHeight );
        }
      }

      return null;
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculateLocalMinimumHeight(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.minimumHeight !== null ) {
            return Math.abs( this.transform.inverseDeltaY( this.minimumHeight ) );
          }
        }
        else if ( this.minimumWidth !== null ) {
          return Math.abs( this.transform.getInverse().m10() * this.minimumWidth );
        }
      }

      return null;
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculateMinimumWidth(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.localMinimumWidth !== null ) {
            return Math.abs( this.transform.transformDeltaX( this.localMinimumWidth ) );
          }
        }
        else if ( this.localMinimumHeight !== null ) {
          return Math.abs( this.transform.matrix.m01() * this.localMinimumHeight );
        }
      }

      return null;
    }

    // Override the calculation to potentially include the opposite dimension (if we have a rotation of that type)
    protected override _calculateMinimumHeight(): number | null {
      if ( this.matrix.isAxisAligned() ) {
        if ( this.matrix.isAligned() ) {
          if ( this.localMinimumHeight !== null ) {
            return Math.abs( this.transform.transformDeltaY( this.localMinimumHeight ) );
          }
        }
        else if ( this.localMinimumWidth !== null ) {
          return Math.abs( this.transform.matrix.m10() * this.localMinimumWidth );
        }
      }

      return null;
    }
  } );

  // If we're extending into a Node type, include option keys
  if ( SuperExtendedType.prototype._mutatorKeys ) {
    const existingKeys = SuperExtendedType.prototype._mutatorKeys;
    const newKeys = SIZABLE_SELF_OPTION_KEYS;
    const indexOfBoundsBasedOptions = existingKeys.indexOf( REQUIRES_BOUNDS_OPTION_KEYS[ 0 ] );
    SizableTrait.prototype._mutatorKeys = [
      ...existingKeys.slice( 0, indexOfBoundsBasedOptions ),
      ...newKeys,
      ...existingKeys.slice( indexOfBoundsBasedOptions )
    ];
  }

  return SizableTrait;
} );

// Some typescript gymnastics to provide a user-defined type guard that treats something as Sizable
// We need to define an unused function with a concrete type, so that we can extract the return type of the function
// and provide a type for a Node that extends this type.
const wrapper = () => Sizable( Node );
export type SizableNode = InstanceType<ReturnType<typeof wrapper>>;

const isSizable = ( node: Node ): node is SizableNode => {
  return node.widthSizable && node.heightSizable;
};
const extendsSizable = ( node: Node ): node is SizableNode => {
  return node.extendsSizable;
};

scenery.register( 'Sizable', Sizable );
export default Sizable;
export { isSizable, extendsSizable };
