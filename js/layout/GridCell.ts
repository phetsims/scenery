// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for GridConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import { GridConfigurable, GridConfigurableOptions, GridConstraint, LayoutProxy, Node, scenery, TrackingLayoutProxyProperty } from '../imports.js';
import { GridConfigurableAlign } from './GridConfigurable.js';
import optionize from '../../../phet-core/js/optionize.js';

const sizableFlagPair = new OrientationPair( 'widthSizable' as const, 'heightSizable' as const );
const minimumSizePair = new OrientationPair( 'minimumWidth' as const, 'minimumHeight' as const );
const preferredSizePair = new OrientationPair( 'preferredWidth' as const, 'preferredHeight' as const );

// {number} - Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

type SelfOptions = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

export type GridCellOptions = SelfOptions & GridConfigurableOptions;

export default class GridCell extends GridConfigurable( Object ) {

  private readonly _constraint: GridConstraint;
  private readonly _node: Node;
  private _proxy: LayoutProxy | null;
  private readonly layoutOptionsListener: () => void;
  private readonly layoutProxyProperty: TrackingLayoutProxyProperty | null;

  // These are only set initially, and ignored for the future
  position: OrientationPair<number>;
  size: OrientationPair<number>;

  // Set to be the bounds available for the cell
  lastAvailableBounds: Bounds2;

  // Set to be the bounds used by the cell
  lastUsedBounds: Bounds2;

  constructor( constraint: GridConstraint, node: Node, proxy: LayoutProxy | null ) {

    const options = optionize<GridCellOptions, SelfOptions, GridConfigurableOptions>()( {
      x: 0,
      y: 0,
      width: 1,
      height: 1
    }, node.layoutOptions as GridConfigurableOptions );

    assert && assert( typeof options.x === 'number' && Number.isInteger( options.x ) && isFinite( options.x ) && options.x >= 0 );
    assert && assert( typeof options.y === 'number' && Number.isInteger( options.y ) && isFinite( options.y ) && options.y >= 0 );
    assert && assert( typeof options.width === 'number' && Number.isInteger( options.width ) && isFinite( options.width ) && options.width >= 1 );
    assert && assert( typeof options.height === 'number' && Number.isInteger( options.height ) && isFinite( options.height ) && options.height >= 1 );

    super();

    if ( proxy ) {
      this.layoutProxyProperty = null;
    }
    else {
      // If a LayoutProxy is not provided, we'll listen to (a) all the trails between our ancestor and this node,
      // (b) construct layout proxies for it (and assign here), and (c) listen to ancestor transforms to refresh
      // the layout when needed.
      this.layoutProxyProperty = new TrackingLayoutProxyProperty( constraint.ancestorNode, node, () => constraint.updateLayoutAutomatically() );
      this.layoutProxyProperty.link( proxy => {
        this._proxy = proxy;
      } );
    }

    this._constraint = constraint;
    this._node = node;
    this._proxy = constraint.createLayoutProxy( node )!; // TODO: handle disconnected, and listen for if we disconnect
    this.position = new OrientationPair( options.x, options.y );
    this.size = new OrientationPair( options.width, options.height );
    this.lastAvailableBounds = Bounds2.NOTHING.copy();
    this.lastUsedBounds = Bounds2.NOTHING.copy();

    this.setOptions( options );
    this.onLayoutOptionsChange();

    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );
    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  get effectiveXAlign(): GridConfigurableAlign {
    return this._xAlign !== null ? this._xAlign : this._constraint._xAlign!;
  }

  get effectiveYAlign(): GridConfigurableAlign {
    return this._yAlign !== null ? this._yAlign : this._constraint._yAlign!;
  }

  getEffectiveAlign( orientation: Orientation ): GridConfigurableAlign {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXAlign : this.effectiveYAlign;
  }

  get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._constraint._leftMargin!;
  }

  get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._constraint._rightMargin!;
  }

  get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._constraint._topMargin!;
  }

  get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._constraint._bottomMargin!;
  }

  getEffectiveMinMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveLeftMargin : this.effectiveTopMargin;
  }

  getEffectiveMaxMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveRightMargin : this.effectiveBottomMargin;
  }

  get effectiveXGrow(): number {
    return this._xGrow !== null ? this._xGrow : this._constraint._xGrow!;
  }

  get effectiveYGrow(): number {
    return this._yGrow !== null ? this._yGrow : this._constraint._yGrow!;
  }

  getEffectiveGrow( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXGrow : this.effectiveYGrow;
  }

  get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._constraint._minContentWidth;
  }

  get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._constraint._minContentHeight;
  }

  getEffectiveMinContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMinContentWidth : this.effectiveMinContentHeight;
  }


  get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._constraint._maxContentWidth;
  }

  get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._constraint._maxContentHeight;
  }

  getEffectiveMaxContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMaxContentWidth : this.effectiveMaxContentHeight;
  }

  private onLayoutOptionsChange(): void {
    this.setOptions( this.node.layoutOptions as GridConfigurableOptions );
  }

  private setOptions( options?: GridConfigurableOptions ): void {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }

  get node(): Node {
    return this._node;
  }

  isConnected(): boolean {
    return this._proxy !== null;
  }

  get proxy(): LayoutProxy {
    assert && assert( this._proxy );

    return this._proxy!;
  }

  getMinimumSize( orientation: Orientation ): number {
    return this.getEffectiveMinMargin( orientation ) +
           Math.max(
             this.proxy[ sizableFlagPair.get( orientation ) ] ? this.proxy[ minimumSizePair.get( orientation ) ] || 0 : this.proxy[ orientation.size ],
             this.getEffectiveMinContent( orientation ) || 0
           ) +
           this.getEffectiveMaxMargin( orientation );
  }

  getMaximumSize( orientation: Orientation ): number {
    return this.getEffectiveMinMargin( orientation ) +
           Math.min(
             this.getEffectiveMaxContent( orientation ) || Number.POSITIVE_INFINITY
           ) +
           this.getEffectiveMaxMargin( orientation );
  }

  attemptPreferredSize( orientation: Orientation, value: number ): void {
    if ( this.proxy[ sizableFlagPair.get( orientation ) ] ) {
      const minimumSize = this.getMinimumSize( orientation );
      const maximumSize = this.getMaximumSize( orientation );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      this.proxy[ preferredSizePair.get( orientation ) ] = value - this.getEffectiveMinMargin( orientation ) - this.getEffectiveMaxMargin( orientation );
    }
  }

  attemptPosition( orientation: Orientation, align: GridConfigurableAlign, value: number, availableSize: number ): void {
    if ( align === GridConfigurableAlign.ORIGIN ) {
      // TODO: handle layout bounds
      // TODO: OMG this is horribly broken right? We would need to align stuff first
      // TODO: Do a pass to handle origin cells first (and in FLOW too)
      if ( Math.abs( this.proxy[ orientation.coordinate ] - value ) > CHANGE_POSITION_THRESHOLD ) {
        this.proxy[ orientation.coordinate ] = value;
      }
    }
    else {
      const minMargin = this.getEffectiveMinMargin( orientation );
      const maxMargin = this.getEffectiveMaxMargin( orientation );
      const extraSpace = availableSize - this.proxy[ orientation.size ] - minMargin - maxMargin;
      value += minMargin + extraSpace * align.padRatio;

      if ( Math.abs( this.proxy[ orientation.minSide ] - value ) > CHANGE_POSITION_THRESHOLD ) {
        this.proxy[ orientation.minSide ] = value;
      }
    }
  }

  getCellBounds(): Bounds2 {
    return this.proxy.bounds.withOffsets(
      this.effectiveLeftMargin,
      this.effectiveTopMargin,
      this.effectiveRightMargin,
      this.effectiveBottomMargin );
  }

  containsIndex( orientation: Orientation, index: number ): boolean {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return index >= position && index < position + size;
  }

  containsRow( row: number ): boolean {
    return this.containsIndex( Orientation.VERTICAL, row );
  }

  containsColumn( column: number ): boolean {
    return this.containsIndex( Orientation.HORIZONTAL, column );
  }

  getIndices( orientation: Orientation ): number[] {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return _.range( position, position + size );
  }

  /**
   * Releases references
   */
  dispose(): void {
    this.layoutProxyProperty && this.layoutProxyProperty.dispose();

    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'GridCell', GridCell );
