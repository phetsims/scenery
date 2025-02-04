// Copyright 2021-2025, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for GridConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import optionize from '../../../../phet-core/js/optionize.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../../phet-core/js/OrientationPair.js';
import type { ExternalGridConfigurableOptions } from '../../layout/constraints/GridConfigurable.js';
import { GRID_CONFIGURABLE_OPTION_KEYS } from '../../layout/constraints/GridConfigurable.js';
import GridConfigurable from '../../layout/constraints/GridConfigurable.js';
import type GridConstraint from '../../layout/constraints/GridConstraint.js';
import LayoutAlign from '../../layout/LayoutAlign.js';
import type LayoutProxy from '../../layout/LayoutProxy.js';
import MarginLayoutCell from '../../layout/constraints/MarginLayoutCell.js';
import Node from '../../nodes/Node.js';
import scenery from '../../scenery.js';

const GRID_CELL_KEYS = [
  ...GRID_CONFIGURABLE_OPTION_KEYS,
  'row',
  'column',
  'horizontalSpan',
  'verticalSpan'
];

type SelfOptions = {

  // Defines the row (or if height>1, the top-most row) index of the cell. row:0 is the top-most row
  row?: number;

  // Defines the column (or if width>1, the left-most column) index of the cell. column:0 is the left-most column.
  column?: number;

  // How many columns this one cell spans.
  horizontalSpan?: number;

  // How many rows this one cell spans
  verticalSpan?: number;
};

export type GridCellOptions = SelfOptions & ExternalGridConfigurableOptions;

export default class GridCell extends GridConfigurable( MarginLayoutCell ) {

  // (scenery-internal) These are only set initially, and ignored for the future
  public position!: OrientationPair<number>;
  public size!: OrientationPair<number>;

  private readonly gridConstraint: GridConstraint;

  /**
   * (scenery-internal)
   */
  public constructor( constraint: GridConstraint, node: Node, proxy: LayoutProxy | null ) {

    super( constraint, node, proxy );

    this.gridConstraint = constraint;

    this.setOptions( node.layoutOptions as ExternalGridConfigurableOptions );
    this.onLayoutOptionsChange( true );
  }

  /**
   * Return the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveXAlign(): LayoutAlign {
    return this._xAlign !== null ? this._xAlign : this.gridConstraint._xAlign!;
  }

  /**
   * Return the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveYAlign(): LayoutAlign {
    return this._yAlign !== null ? this._yAlign : this.gridConstraint._yAlign!;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveAlign( orientation: Orientation ): LayoutAlign {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXAlign : this.effectiveYAlign;
  }

  /**
   * Return the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveXGrow(): number {
    return this._xGrow !== null ? this._xGrow : this.gridConstraint._xGrow!;
  }

  /**
   * Return the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveYGrow(): number {
    return this._yGrow !== null ? this._yGrow : this.gridConstraint._yGrow!;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveGrow( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXGrow : this.effectiveYGrow;
  }

  /**
   * Return the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveXStretch(): boolean {
    return this._xStretch !== null ? this._xStretch : this.gridConstraint._xStretch!;
  }

  /**
   * Return the used value, with this cell's value taking precedence over the constraint's default
   * (scenery-internal)
   */
  public get effectiveYStretch(): boolean {
    return this._yStretch !== null ? this._yStretch : this.gridConstraint._yStretch!;
  }

  /**
   * (scenery-internal)
   */
  public getEffectiveStretch( orientation: Orientation ): boolean {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXStretch : this.effectiveYStretch;
  }

  protected override onLayoutOptionsChange( isInitial = false ): void {
    this.setOptions( this.node.layoutOptions as ExternalGridConfigurableOptions );

    super.onLayoutOptionsChange();

    if ( !isInitial ) {
      this.gridConstraint.updateLayoutAutomatically();
    }
  }

  private setOptions( providedOptions?: ExternalGridConfigurableOptions ): void {

    // We'll have defaults for cells (the horizontalSpan/verticalSpan are especially relevant)
    const options = optionize<GridCellOptions, SelfOptions, ExternalGridConfigurableOptions>()( {
      column: 0,
      row: 0,
      horizontalSpan: 1,
      verticalSpan: 1
    }, providedOptions );

    assert && Object.keys( options ).forEach( key => {
      assert && assert( GRID_CELL_KEYS.includes( key ) || key === 'isSeparator',
        `Cannot provide key ${key} to a GridCell's layoutOptions. Perhaps this is a Flow-style layout option?` );
    } );

    assert && assert( typeof options.column === 'number' && Number.isInteger( options.column ) && isFinite( options.column ) && options.column >= 0 );
    assert && assert( typeof options.row === 'number' && Number.isInteger( options.row ) && isFinite( options.row ) && options.row >= 0 );
    assert && assert( typeof options.horizontalSpan === 'number' && Number.isInteger( options.horizontalSpan ) && isFinite( options.horizontalSpan ) && options.horizontalSpan >= 1 );
    assert && assert( typeof options.verticalSpan === 'number' && Number.isInteger( options.verticalSpan ) && isFinite( options.verticalSpan ) && options.verticalSpan >= 1 );

    // Clear options that are not provided in the mutate options (but if they are present, do NOT clear, so we avoid
    // more potential for infinite loops)
    this.setConfigToInherit( options );

    this.position = new OrientationPair( options.column, options.row );
    this.size = new OrientationPair( options.horizontalSpan, options.verticalSpan );

    this.mutateConfigurable( options );
  }

  /**
   * Whether this cell contains the given row/column (based on the orientation). Due to horizontalSpan/verticalSpan of the cell,
   * this could be true for multiple indices.
   * (scenery-internal)
   */
  public containsIndex( orientation: Orientation, index: number ): boolean {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return index >= position && index < position + size;
  }

  /**
   * Whether this cell contains the given row.
   * (scenery-internal)
   */
  public containsRow( row: number ): boolean {
    return this.containsIndex( Orientation.VERTICAL, row );
  }

  /**
   * Whether this cell contains the given column.
   * (scenery-internal)
   */
  public containsColumn( column: number ): boolean {
    return this.containsIndex( Orientation.HORIZONTAL, column );
  }

  /**
   * Returns the row/column indices that this cell spans (based on the orientation)
   * (scenery-internal)
   */
  public getIndices( orientation: Orientation ): number[] {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return _.range( position, position + size );
  }
}

scenery.register( 'GridCell', GridCell );