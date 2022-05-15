// Copyright 2022, University of Colorado Boulder

/**
 * Supertype for LayoutConstraints that are based on an actual Node where the layout takes place. Generally used with
 * layout containers that are subtypes of LayoutNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import { LayoutConstraint, LayoutProxy, Node, scenery } from '../imports.js';
import IProperty from '../../../axon/js/IProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import Tandem from '../../../tandem/js/Tandem.js';
import Vector2 from '../../../dot/js/Vector2.js';
import Orientation from '../../../phet-core/js/Orientation.js';

// Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

type SelfOptions = {
  // Whether invisible Nodes are excluded from the layout.
  excludeInvisible?: boolean;

  // If available, the local versions of these Properties for the layout container should be passed in. We do the
  // layout in the local coordinate frame of e.g. GridBox/FlowBox. It's named this way just for ease-of-use within
  // this code.
  preferredWidthProperty?: IProperty<number | null>;
  preferredHeightProperty?: IProperty<number | null>;
  minimumWidthProperty?: IProperty<number | null>;
  minimumHeightProperty?: IProperty<number | null>;

  // If provided, will position content at an offset from the normal origin
  layoutOriginProperty?: IProperty<Vector2>;
};

export type NodeLayoutConstraintOptions = SelfOptions;

// Type export designed for use with clients
export type NodeLayoutAvailableConstraintOptions = Pick<NodeLayoutConstraintOptions, 'excludeInvisible' | 'layoutOriginProperty'>;

export default class NodeLayoutConstraint extends LayoutConstraint {

  private _excludeInvisible = true;

  // Reports out the used layout bounds (may be larger than actual bounds, since it will include margins, etc.)
  // Layout nodes can use this to adjust their localBounds
  readonly layoutBoundsProperty: IProperty<Bounds2>;

  readonly preferredWidthProperty: IProperty<number | null>;
  readonly preferredHeightProperty: IProperty<number | null>;
  readonly minimumWidthProperty: IProperty<number | null>;
  readonly minimumHeightProperty: IProperty<number | null>;
  readonly layoutOriginProperty: IProperty<Vector2>;

  // Recommended for ancestorNode to be the layout container, and that the layout container extends LayoutNode.
  constructor( ancestorNode: Node, providedOptions?: NodeLayoutConstraintOptions ) {
    assert && assert( ancestorNode instanceof Node );

    // The omitted options are set to proper defaults below
    const options = optionize<NodeLayoutConstraintOptions, Omit<SelfOptions, 'excludeInvisible' | 'spacing' | 'xSpacing' | 'ySpacing'>, {}>()( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty<number | null>( null ),
      preferredHeightProperty: new TinyProperty<number | null>( null ),
      minimumWidthProperty: new TinyProperty<number | null>( null ),
      minimumHeightProperty: new TinyProperty<number | null>( null ),
      layoutOriginProperty: new TinyProperty<Vector2>( Vector2.ZERO )
    }, providedOptions );

    super( ancestorNode );

    this.layoutBoundsProperty = new Property( Bounds2.NOTHING, {
      useDeepEquality: true,
      tandem: Tandem.OPT_OUT
    } );

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;
    this.layoutOriginProperty = options.layoutOriginProperty;

    this.preferredWidthProperty.lazyLink( this._updateLayoutListener );
    this.preferredHeightProperty.lazyLink( this._updateLayoutListener );
    this.layoutOriginProperty.lazyLink( this._updateLayoutListener );
  }

  get excludeInvisible(): boolean {
    return this._excludeInvisible;
  }

  set excludeInvisible( value: boolean ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._excludeInvisible !== value ) {
      this._excludeInvisible = value;

      this.updateLayoutAutomatically();
    }
  }

  setProxyPreferredSize( orientation: Orientation, proxy: LayoutProxy, preferredSize: number ): void {
    proxy[ orientation.preferredSize ] = preferredSize;
  }

  setProxyMinSide( orientation: Orientation, proxy: LayoutProxy, minSide: number ): void {
    if ( Math.abs( proxy[ orientation.minSide ] - minSide ) > CHANGE_POSITION_THRESHOLD ) {
      proxy[ orientation.minSide ] = minSide;
    }
  }

  setProxyOrigin( orientation: Orientation, proxy: LayoutProxy, origin: number ): void {
    if ( Math.abs( proxy[ orientation.coordinate ] - origin ) > CHANGE_POSITION_THRESHOLD ) {
      proxy[ orientation.coordinate ] = origin;
    }
  }

  /**
   * Releases references
   */
  override dispose(): void {
    // In case they're from external sources (since these constraints can be used without a dedicated Node that is also
    // being disposed.
    this.preferredWidthProperty.unlink( this._updateLayoutListener );
    this.preferredHeightProperty.unlink( this._updateLayoutListener );
    this.layoutOriginProperty.unlink( this._updateLayoutListener );

    super.dispose();
  }
}

scenery.register( 'NodeLayoutConstraint', NodeLayoutConstraint );
