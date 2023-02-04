// Copyright 2022-2023, University of Colorado Boulder

/**
 * Supertype for LayoutConstraints that are based on an actual Node where the layout takes place. Generally used with
 * layout containers that are subtypes of LayoutNode.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../../axon/js/Property.js';
import StrictOmit from '../../../../phet-core/js/types/StrictOmit.js';
import TinyProperty from '../../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../../dot/js/Bounds2.js';
import { LayoutConstraint, LayoutProxy, MarginLayoutCell, Node, scenery } from '../../imports.js';
import TProperty from '../../../../axon/js/TProperty.js';
import optionize from '../../../../phet-core/js/optionize.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Orientation from '../../../../phet-core/js/Orientation.js';

// Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

type SelfOptions = {
  // Whether invisible Nodes are excluded from the layout.
  excludeInvisible?: boolean;

  // If available, the local versions of these Properties for the layout container should be passed in. We do the
  // layout in the local coordinate frame of e.g. GridBox/FlowBox. It's named this way just for ease-of-use within
  // this code.
  preferredWidthProperty?: TProperty<number | null>;
  preferredHeightProperty?: TProperty<number | null>;
  minimumWidthProperty?: TProperty<number | null>;
  minimumHeightProperty?: TProperty<number | null>;

  // If provided, will position content at an offset from the normal origin
  layoutOriginProperty?: TProperty<Vector2>;
};

export type NodeLayoutConstraintOptions = SelfOptions;

// Type export designed for use with clients
export type NodeLayoutAvailableConstraintOptions = Pick<NodeLayoutConstraintOptions, 'excludeInvisible' | 'layoutOriginProperty'>;

export default class NodeLayoutConstraint extends LayoutConstraint {

  private _excludeInvisible = true;

  // Reports out the used layout bounds (may be larger than actual bounds, since it will include margins, etc.)
  // Layout nodes can use this to adjust their localBounds. FlowBox/GridBox uses this for their localBounds.
  // (scenery-internal)
  public readonly layoutBoundsProperty: TProperty<Bounds2>;

  // (scenery-internal)
  public readonly preferredWidthProperty: TProperty<number | null>;
  public readonly preferredHeightProperty: TProperty<number | null>;
  public readonly minimumWidthProperty: TProperty<number | null>;
  public readonly minimumHeightProperty: TProperty<number | null>;
  public readonly layoutOriginProperty: TProperty<Vector2>;

  /**
   * Recommended for ancestorNode to be the layout container, and that the layout container extends LayoutNode.
   * (scenery-internal)
   */
  public constructor( ancestorNode: Node, providedOptions?: NodeLayoutConstraintOptions ) {

    // The omitted options are set to proper defaults below
    const options = optionize<NodeLayoutConstraintOptions, StrictOmit<SelfOptions, 'excludeInvisible'>>()( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty<number | null>( null ),
      preferredHeightProperty: new TinyProperty<number | null>( null ),
      minimumWidthProperty: new TinyProperty<number | null>( null ),
      minimumHeightProperty: new TinyProperty<number | null>( null ),
      layoutOriginProperty: new TinyProperty<Vector2>( Vector2.ZERO )
    }, providedOptions );

    super( ancestorNode );

    this.layoutBoundsProperty = new Property( Bounds2.NOTHING, {
      valueComparisonStrategy: 'equalsFunction'
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

  /**
   * Filters out cells to only those that will be involved in layout
   */
  protected filterLayoutCells<Cell extends MarginLayoutCell>( cells: Cell[] ): Cell[] {
    // We'll check to make sure cells are disposed in a common place, so it's not duplicated
    assert && assert( _.every( cells, cell => !cell.node.isDisposed ), 'A cell\'s node should not be disposed when layout happens' );

    return cells.filter( cell => {
      return cell.isConnected() && cell.proxy.bounds.isValid() && ( !this.excludeInvisible || cell.node.visible );
    } );
  }

  public get excludeInvisible(): boolean {
    return this._excludeInvisible;
  }

  public set excludeInvisible( value: boolean ) {
    if ( this._excludeInvisible !== value ) {
      this._excludeInvisible = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * Sets preferred size of content in a central location (so we could hook in animation in the future)
   * (scenery-internal)
   */
  public setProxyPreferredSize( orientation: Orientation, proxy: LayoutProxy, preferredSize: number | null ): void {
    proxy[ orientation.preferredSize ] = preferredSize;
  }

  /**
   * Sets position of content in a central location (so we could hook in animation in the future)
   * (scenery-internal)
   */
  public setProxyMinSide( orientation: Orientation, proxy: LayoutProxy, minSide: number ): void {
    if ( Math.abs( proxy[ orientation.minSide ] - minSide ) > CHANGE_POSITION_THRESHOLD ) {
      proxy[ orientation.minSide ] = minSide;
    }
  }

  /**
   * Sets origin-based position of content in a central location (so we could hook in animation in the future)
   * (scenery-internal)
   */
  public setProxyOrigin( orientation: Orientation, proxy: LayoutProxy, origin: number ): void {
    if ( Math.abs( proxy[ orientation.coordinate ] - origin ) > CHANGE_POSITION_THRESHOLD ) {
      proxy[ orientation.coordinate ] = origin;
    }
  }

  /**
   * Releases references
   */
  public override dispose(): void {
    // In case they're from external sources (since these constraints can be used without a dedicated Node that is also
    // being disposed.
    this.preferredWidthProperty.unlink( this._updateLayoutListener );
    this.preferredHeightProperty.unlink( this._updateLayoutListener );
    this.layoutOriginProperty.unlink( this._updateLayoutListener );

    super.dispose();
  }
}

scenery.register( 'NodeLayoutConstraint', NodeLayoutConstraint );
