// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for GridConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../../dot/js/Bounds2.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../../phet-core/js/OrientationPair.js';
import { ExternalGridConfigurableOptions, GridConfigurable, GridConstraint, LayoutAlign, LayoutProxy, MarginLayoutCell, Node, scenery } from '../../imports.js';
import optionize from '../../../../phet-core/js/optionize.js';

type SelfOptions = {
  // Defines the column (or if width>1, the left-most column) index of the cell. x:0 is the left-most column.
  x?: number;

  // Defines the row (or if height>1, the top-most row) index of the cell. y:0 is the top-most row
  y?: number;

  // How many columns this one cell spans.
  width?: number;

  // How many rows this one cell spans
  height?: number;
};

export type GridCellOptions = SelfOptions & ExternalGridConfigurableOptions;

export default class GridCell extends GridConfigurable( MarginLayoutCell ) {

  // These are only set initially, and ignored for the future
  position!: OrientationPair<number>;
  size!: OrientationPair<number>;

  // Set to be the bounds available for the cell
  lastAvailableBounds: Bounds2 = Bounds2.NOTHING.copy();

  // Set to be the bounds used by the cell
  lastUsedBounds: Bounds2 = Bounds2.NOTHING.copy();

  private readonly gridConstraint: GridConstraint;

  constructor( constraint: GridConstraint, node: Node, proxy: LayoutProxy | null ) {

    super( constraint, node, proxy );

    this.gridConstraint = constraint;

    this.setOptions( node.layoutOptions as ExternalGridConfigurableOptions );
    this.onLayoutOptionsChange();
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  get effectiveXAlign(): LayoutAlign {
    return this._xAlign !== null ? this._xAlign : this.gridConstraint._xAlign!;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  get effectiveYAlign(): LayoutAlign {
    return this._yAlign !== null ? this._yAlign : this.gridConstraint._yAlign!;
  }

  getEffectiveAlign( orientation: Orientation ): LayoutAlign {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXAlign : this.effectiveYAlign;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  get effectiveXGrow(): number {
    return this._xGrow !== null ? this._xGrow : this.gridConstraint._xGrow!;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  get effectiveYGrow(): number {
    return this._yGrow !== null ? this._yGrow : this.gridConstraint._yGrow!;
  }

  getEffectiveGrow( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXGrow : this.effectiveYGrow;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  get effectiveXStretch(): boolean {
    return this._xStretch !== null ? this._xStretch : this.gridConstraint._xStretch!;
  }

  // The used value, with this cell's value taking precedence over the constraint's default
  get effectiveYStretch(): boolean {
    return this._yStretch !== null ? this._yStretch : this.gridConstraint._yStretch!;
  }

  getEffectiveStretch( orientation: Orientation ): boolean {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXStretch : this.effectiveYStretch;
  }

  protected override onLayoutOptionsChange(): void {
    this.setOptions( this.node.layoutOptions as ExternalGridConfigurableOptions );

    super.onLayoutOptionsChange();
  }

  private setOptions( providedOptions?: ExternalGridConfigurableOptions ): void {

    // We'll have defaults for cells (the width/height are especially relevant)
    const options = optionize<GridCellOptions, SelfOptions, ExternalGridConfigurableOptions>()( {
      x: 0,
      y: 0,
      width: 1,
      height: 1
    }, providedOptions );

    assert && assert( typeof options.x === 'number' && Number.isInteger( options.x ) && isFinite( options.x ) && options.x >= 0 );
    assert && assert( typeof options.y === 'number' && Number.isInteger( options.y ) && isFinite( options.y ) && options.y >= 0 );
    assert && assert( typeof options.width === 'number' && Number.isInteger( options.width ) && isFinite( options.width ) && options.width >= 1 );
    assert && assert( typeof options.height === 'number' && Number.isInteger( options.height ) && isFinite( options.height ) && options.height >= 1 );

    this.setConfigToInherit();

    this.position = new OrientationPair( options.x, options.y );
    this.size = new OrientationPair( options.width, options.height );

    this.mutateConfigurable( options );
  }

  // Whether this cell contains the given row/column (based on the orientation). Due to width/height of the cell,
  // this could be true for multiple indices.
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

  // Returns the row/column indices that this cell spans (based on the orientation)
  getIndices( orientation: Orientation ): number[] {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return _.range( position, position + size );
  }
}

scenery.register( 'GridCell', GridCell );
