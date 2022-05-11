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
import { GridConfigurable, GridConfigurableOptions, GridConstraint, LayoutProxy, Node, scenery } from '../imports.js';
import { GridConfigurableAlign } from './GridConfigurable.js';
import optionize from '../../../phet-core/js/optionize.js';
import LayoutCell from './LayoutCell.js';

const sizableFlagPair = new OrientationPair( 'widthSizable' as const, 'heightSizable' as const );
const minimumSizePair = new OrientationPair( 'minimumWidth' as const, 'minimumHeight' as const );
const preferredSizePair = new OrientationPair( 'preferredWidth' as const, 'preferredHeight' as const );

// Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

type SelfOptions = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

export type GridCellOptions = SelfOptions & GridConfigurableOptions;

export default class GridCell extends GridConfigurable( LayoutCell ) {

  // These are only set initially, and ignored for the future
  position: OrientationPair<number>;
  size: OrientationPair<number>;

  // Set to be the bounds available for the cell
  lastAvailableBounds: Bounds2;

  // Set to be the bounds used by the cell
  lastUsedBounds: Bounds2;

  private _gridConstraint: GridConstraint;

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

    super( constraint, node, proxy );

    this._gridConstraint = constraint;

    this.position = new OrientationPair( options.x, options.y );
    this.size = new OrientationPair( options.width, options.height );
    this.lastAvailableBounds = Bounds2.NOTHING.copy();
    this.lastUsedBounds = Bounds2.NOTHING.copy();

    this.setOptions( options );
    this.onLayoutOptionsChange();
  }

  get effectiveXAlign(): GridConfigurableAlign {
    return this._xAlign !== null ? this._xAlign : this._gridConstraint._xAlign!;
  }

  get effectiveYAlign(): GridConfigurableAlign {
    return this._yAlign !== null ? this._yAlign : this._gridConstraint._yAlign!;
  }

  getEffectiveAlign( orientation: Orientation ): GridConfigurableAlign {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXAlign : this.effectiveYAlign;
  }

  get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._gridConstraint._leftMargin!;
  }

  get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._gridConstraint._rightMargin!;
  }

  get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._gridConstraint._topMargin!;
  }

  get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._gridConstraint._bottomMargin!;
  }

  getEffectiveMinMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveLeftMargin : this.effectiveTopMargin;
  }

  getEffectiveMaxMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveRightMargin : this.effectiveBottomMargin;
  }

  get effectiveXGrow(): number {
    return this._xGrow !== null ? this._xGrow : this._gridConstraint._xGrow!;
  }

  get effectiveYGrow(): number {
    return this._yGrow !== null ? this._yGrow : this._gridConstraint._yGrow!;
  }

  getEffectiveGrow( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXGrow : this.effectiveYGrow;
  }

  get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._gridConstraint._minContentWidth;
  }

  get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._gridConstraint._minContentHeight;
  }

  getEffectiveMinContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMinContentWidth : this.effectiveMinContentHeight;
  }

  get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._gridConstraint._maxContentWidth;
  }

  get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._gridConstraint._maxContentHeight;
  }

  getEffectiveMaxContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMaxContentWidth : this.effectiveMaxContentHeight;
  }

  protected override onLayoutOptionsChange(): void {
    this.setOptions( this.node.layoutOptions as GridConfigurableOptions );

    super.onLayoutOptionsChange();
  }

  private setOptions( options?: GridConfigurableOptions ): void {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
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

  // TODO: Create a Cell type that has margins built in, so we can handle these? Combine Flow and Grid code
  positionStart( orientation: Orientation, value: number ): void {
    const start = this.getEffectiveMinMargin( orientation ) + value;

    if ( Math.abs( this.proxy[ orientation.minSide ] - start ) > CHANGE_POSITION_THRESHOLD ) {
      this.proxy[ orientation.minSide ] = start;
    }
  }

  positionOrigin( orientation: Orientation, value: number ): void {
    if ( Math.abs( this.proxy[ orientation.coordinate ] - value ) > CHANGE_POSITION_THRESHOLD ) {
      this.proxy[ orientation.coordinate ] = value;
    }
  }

  /**
   * Returns the bounding box of the cell if it was repositioned to have its origin shifted to the origin of the
   * ancestor node's local coordinate frame.
   */
  getOriginBounds(): Bounds2 {
    return this.getCellBounds().shiftedXY( -this.proxy.x, -this.proxy.y );
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
}

scenery.register( 'GridCell', GridCell );
