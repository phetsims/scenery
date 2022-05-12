// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for GridConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import { GridConfigurable, GridConfigurableOptions, GridConstraint, LayoutAlign, LayoutProxy, MarginLayoutCell, Node, scenery } from '../imports.js';
import optionize from '../../../phet-core/js/optionize.js';

type SelfOptions = {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
};

export type GridCellOptions = SelfOptions & GridConfigurableOptions;

export default class GridCell extends GridConfigurable( MarginLayoutCell ) {

  // These are only set initially, and ignored for the future
  position: OrientationPair<number>;
  size: OrientationPair<number>;

  // Set to be the bounds available for the cell
  lastAvailableBounds: Bounds2;

  // Set to be the bounds used by the cell
  lastUsedBounds: Bounds2;

  gridConstraint: GridConstraint;

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

    this.gridConstraint = constraint;

    this.position = new OrientationPair( options.x, options.y );
    this.size = new OrientationPair( options.width, options.height );
    this.lastAvailableBounds = Bounds2.NOTHING.copy();
    this.lastUsedBounds = Bounds2.NOTHING.copy();

    this.setOptions( options );
    this.onLayoutOptionsChange();
  }

  get effectiveXAlign(): LayoutAlign {
    return this._xAlign !== null ? this._xAlign : this.gridConstraint._xAlign!;
  }

  get effectiveYAlign(): LayoutAlign {
    return this._yAlign !== null ? this._yAlign : this.gridConstraint._yAlign!;
  }

  getEffectiveAlign( orientation: Orientation ): LayoutAlign {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXAlign : this.effectiveYAlign;
  }

  get effectiveXGrow(): number {
    return this._xGrow !== null ? this._xGrow : this.gridConstraint._xGrow!;
  }

  get effectiveYGrow(): number {
    return this._yGrow !== null ? this._yGrow : this.gridConstraint._yGrow!;
  }

  getEffectiveGrow( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXGrow : this.effectiveYGrow;
  }

  get effectiveXStretch(): boolean {
    return this._xStretch !== null ? this._xStretch : this.gridConstraint._xStretch!;
  }

  get effectiveYStretch(): boolean {
    return this._yStretch !== null ? this._yStretch : this.gridConstraint._yStretch!;
  }

  getEffectiveStretch( orientation: Orientation ): boolean {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXStretch : this.effectiveYStretch;
  }

  protected override onLayoutOptionsChange(): void {
    this.setOptions( this.node.layoutOptions as GridConfigurableOptions );

    super.onLayoutOptionsChange();
  }

  private setOptions( options?: GridConfigurableOptions ): void {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
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
