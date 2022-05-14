// Copyright 2021-2022, University of Colorado Boulder

/**
 * Main flow-layout logic. Usually used indirectly through FlowBox, but can also be used directly (say, if nodes don't
 * have the same parent, or a FlowBox can't be used).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import mutate from '../../../phet-core/js/mutate.js';
import { Divider, FLOW_CONFIGURABLE_OPTION_KEYS, FlowCell, FlowConfigurable, FlowConfigurableOptions, FlowLine, HorizontalLayoutJustification, LayoutAlign, LayoutJustification, Node, NodeLayoutAvailableConstraintOptions, NodeLayoutConstraint, scenery, VerticalLayoutJustification } from '../imports.js';
import IProperty from '../../../axon/js/IProperty.js';

const FLOW_CONSTRAINT_OPTION_KEYS = [
  ...FLOW_CONFIGURABLE_OPTION_KEYS,
  'spacing',
  'lineSpacing',
  'justify',
  'justifyLines',
  'wrap',
  'excludeInvisible'
];

type SelfOptions = {
  // The default spacing in-between elements in the primary direction. If additional (or less) spacing is desired for
  // certain elements, per-element margins (even negative) can be set in the layoutOptions of nodes contained.
  spacing?: number;

  // The default spacing in-between lines in the secondary direction.
  lineSpacing?: number;

  // How extra space in the primary direction is allocated. The default is spaceBetween.
  justify?: HorizontalLayoutJustification | VerticalLayoutJustification;

  // How extra space in the secondary direction is allocated. The default is null (which will expand content to fit)
  justifyLines?: HorizontalLayoutJustification | VerticalLayoutJustification | null;

  // Whether line-wrapping is enabled. If so, the primary preferred dimension will determine where things are wrapped.
  wrap?: boolean;

  excludeInvisible?: boolean;

  // The preferred width/height (ideally from a container's localPreferredWidth/localPreferredHeight.
  preferredWidthProperty?: IProperty<number | null>;
  preferredHeightProperty?: IProperty<number | null>;

  // The minimum width/height (ideally from a container's localMinimumWidth/localMinimumHeight.
  minimumWidthProperty?: IProperty<number | null>;
  minimumHeightProperty?: IProperty<number | null>;
};

export type FlowConstraintOptions = SelfOptions & FlowConfigurableOptions & NodeLayoutAvailableConstraintOptions;

export default class FlowConstraint extends FlowConfigurable( NodeLayoutConstraint ) {

  private readonly cells: FlowCell[] = [];
  private _justify: LayoutJustification = LayoutJustification.SPACE_BETWEEN;
  private _justifyLines: LayoutJustification | null = null;
  private _wrap = false;
  private _spacing = 0;
  private _lineSpacing = 0;

  constructor( ancestorNode: Node, providedOptions?: FlowConstraintOptions ) {
    assert && assert( ancestorNode instanceof Node );

    super( ancestorNode, providedOptions );

    this.setConfigToBaseDefault();
    this.mutateConfigurable( providedOptions );
    mutate( this, FLOW_CONSTRAINT_OPTION_KEYS, providedOptions );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );

    this.orientationChangedEmitter.addListener( () => this.cells.forEach( cell => {
      cell.orientation = this.orientation;
    } ) );
  }

  protected override layout(): void {
    super.layout();

    // The orientation along the laid-out lines
    const orientation = this._orientation;

    // The perpendicular orientation, where alignment is handled
    const oppositeOrientation = this._orientation.opposite;

    assert && assert( _.every( this.cells, cell => !cell.node.isDisposed ) );

    // Determine the first index of a visible non-divider (we need to NOT temporarily change visibility of dividers
    // back and forth, since this would trigger recursive layouts). We'll hide dividers until this.
    let firstVisibleNonDividerIndex = 0;
    for ( ; firstVisibleNonDividerIndex < this.cells.length; firstVisibleNonDividerIndex++ ) {
      const cell = this.cells[ firstVisibleNonDividerIndex ];
      if ( cell.node instanceof Divider ) {
        cell.node.visible = false;
      }
      else if ( cell.node.visible ) {
        break;
      }
    }

    // Scan for dividers, toggling visibility as desired. Leave the "last" divider visible, as if they are marking
    // sections "after" themselves.
    let hasVisibleNonDivider = false;
    for ( let i = this.cells.length - 1; i > firstVisibleNonDividerIndex; i-- ) {
      const cell = this.cells[ i ];
      if ( cell.node instanceof Divider ) {
        cell.node.visible = hasVisibleNonDivider;
        hasVisibleNonDivider = false;
      }
      else if ( cell.node.visible ) {
        hasVisibleNonDivider = true;
      }
    }

    const cells: FlowCell[] = this.cells.filter( cell => {
      return cell.isConnected() && cell.proxy.bounds.isValid() && ( !this.excludeInvisible || cell.node.visible );
    } );

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      return;
    }

    // Determine our preferred sizes (they can be null, in which case)
    let preferredSize: number | null = this.getPreferredProperty( orientation ).value;
    const preferredOppositeSize: number | null = this.getPreferredProperty( oppositeOrientation ).value;

    // What is the largest of the minimum sizes of cells (e.g. if we're wrapping, this would be our minimum size)
    const maxMinimumCellSize: number = Math.max( ...cells.map( cell => cell.getMinimumSize( orientation ) || 0 ) );

    if ( maxMinimumCellSize > ( preferredSize || Number.POSITIVE_INFINITY ) ) {
      preferredSize = maxMinimumCellSize;
    }

    // Wrapping all the cells into lines
    const lines: FlowLine[] = [];
    if ( this.wrap ) {
      let currentLineCells: FlowCell[] = [];
      let availableSpace = preferredSize || Number.POSITIVE_INFINITY;

      while ( cells.length ) {
        const cell = cells.shift()!;
        const cellSpace = cell.getMinimumSize( orientation );

        if ( currentLineCells.length === 0 ) {
          currentLineCells.push( cell );
          availableSpace -= cellSpace;
        }
        else if ( this.spacing + cellSpace <= availableSpace + 1e-7 ) {
          currentLineCells.push( cell );
          availableSpace -= this.spacing + cellSpace;
        }
        else {
          lines.push( FlowLine.pool.create( orientation, currentLineCells ) );
          availableSpace = preferredSize || Number.POSITIVE_INFINITY;

          currentLineCells = [ cell ];
          availableSpace -= cellSpace;
        }
      }

      if ( currentLineCells.length ) {
        lines.push( FlowLine.pool.create( orientation, currentLineCells ) );
      }
    }
    else {
      lines.push( FlowLine.pool.create( orientation, cells ) );
    }

    lines.forEach( line => {
      line.cells.forEach( cell => {
        line.min = Math.max( line.min, cell.getMinimumSize( oppositeOrientation ) );
        line.max = Math.min( line.max, cell.getMaximumSize( oppositeOrientation ) );

        // For origin-specified cells, we will record their maximum reach from the origin, so these can be "summed"
        // (since the origin line may end up taking more space).
        if ( cell.effectiveAlign === LayoutAlign.ORIGIN ) {
          const originBounds = cell.getOriginBounds();
          line.minOrigin = Math.min( originBounds[ oppositeOrientation.minCoordinate ], line.minOrigin );
          line.maxOrigin = Math.max( originBounds[ oppositeOrientation.maxCoordinate ], line.maxOrigin );
        }
      } );

      if ( isFinite( line.minOrigin ) && isFinite( line.maxOrigin ) ) {
        line.size = Math.max( line.min, line.maxOrigin - line.minOrigin );
      }
      else {
        line.size = line.min;
      }
    } );

    // Given our wrapped lines, what is our minimum size we could take up?
    const minimumCurrentSize: number = Math.max( ...lines.map( line => line.getMinimumSize( this.spacing ) ) );
    const minimumCurrentOppositeSize = _.sum( lines.map( line => line.size ) ) + ( lines.length - 1 ) * this.lineSpacing;

    // Used for determining our "minimum" size for preferred sizes... if wrapping is enabled, we can be smaller than
    // current minimums
    const minimumAllowableSize = this.wrap ? maxMinimumCellSize : minimumCurrentSize;

    // Increase things if our preferred size is larger than our minimums (we'll figure out how to compensate
    // for the extra space below).
    const size = Math.max( minimumCurrentSize, preferredSize || 0 );
    const oppositeSize = Math.max( minimumCurrentOppositeSize, preferredOppositeSize || 0 );

    // Primary-direction layout
    lines.forEach( line => {
      const minimumContent = _.sum( line.cells.map( cell => cell.getMinimumSize( orientation ) ) );
      const spacingAmount = this.spacing * ( line.cells.length - 1 );
      let spaceRemaining = size - minimumContent - spacingAmount;

      // Initial pending sizes
      line.cells.forEach( cell => {
        cell._pendingSize = cell.getMinimumSize( orientation );
      } );

      // Grow potential sizes if possible
      let growableCells;
      while ( spaceRemaining > 1e-7 && ( growableCells = line.cells.filter( cell => {
        // Can the cell grow more?
        return cell.effectiveGrow !== 0 && cell._pendingSize < cell.getMaximumSize( orientation ) - 1e-7;
      } ) ).length ) {
        // Total sum of "grow" values in cells that could potentially grow
        const totalGrow = _.sum( growableCells.map( cell => cell.effectiveGrow ) );
        const amountToGrow = Math.min(
          // Smallest amount that any of the cells couldn't grow past (note: proportional to effectiveGrow)
          Math.min( ...growableCells.map( cell => ( cell.getMaximumSize( orientation ) - cell._pendingSize ) / cell.effectiveGrow ) ),

          // Amount each cell grows if all of our extra space fits in ALL the cells
          spaceRemaining / totalGrow
        );

        assert && assert( amountToGrow > 1e-11 );

        growableCells.forEach( cell => {
          cell._pendingSize += amountToGrow * cell.effectiveGrow!;
        } );
        spaceRemaining -= amountToGrow * totalGrow;
      }

      // Update preferred dimension based on the pending size
      line.cells.forEach( cell => cell.attemptPreferredSize( orientation, cell._pendingSize ) );

      const primarySpacingFunction = this._justify.spacingFunctionFactory( spaceRemaining, line.cells.length );

      let position = 0;

      line.cells.forEach( ( cell, index ) => {
        position += primarySpacingFunction( index );
        if ( index > 0 ) {
          position += this.spacing;
        }
        cell.positionStart( orientation, position );
        position += cell._pendingSize;
      } );
    } );

    // Secondary-direction layout
    const oppositeSpaceRemaining = oppositeSize - minimumCurrentOppositeSize;
    const initialOppositePosition = lines[ 0 ].hasOrigin() ? lines[ 0 ].minOrigin : 0;
    let oppositePosition = initialOppositePosition;
    if ( this._justifyLines === null ) {
      // null justifyLines will result in expanding all of our lines into the remaining space.

      // Add space remaining evenly (for now) since we don't have any grow values
      lines.forEach( line => {
        line.size += oppositeSpaceRemaining / lines.length;
      } );

      // Position the lines
      lines.forEach( line => {
        line.position = oppositePosition;
        oppositePosition += line.size + this.lineSpacing;
      } );
    }
    else {
      // If we're justifying lines, we won't add any additional space into things
      const spacingFunction = this._justifyLines.spacingFunctionFactory( oppositeSpaceRemaining, lines.length );

      lines.forEach( ( line, index ) => {
        oppositePosition += spacingFunction( index );
        line.position = oppositePosition;
        oppositePosition += line.size + this.lineSpacing;
      } );
    }
    lines.forEach( line => line.cells.forEach( cell => {
      cell.reposition( oppositeOrientation, line.size, line.position, cell.effectiveStretch, -line.minOrigin, cell.effectiveAlign );
    } ) );

    const minCoordinate = 0;
    const maxCoordinate = size;
    const minOppositeCoordinate = initialOppositePosition;
    const maxOppositeCoordinate = initialOppositePosition + oppositeSize;

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = orientation === Orientation.HORIZONTAL ? new Bounds2(
      minCoordinate,
      minOppositeCoordinate,
      maxCoordinate,
      maxOppositeCoordinate
    ) : new Bounds2(
      minOppositeCoordinate,
      minCoordinate,
      maxOppositeCoordinate,
      maxCoordinate
    );

    // Tell others about our new "minimum" sizes
    this.minimumWidthProperty.value = orientation === Orientation.HORIZONTAL ? minimumAllowableSize : minimumCurrentOppositeSize;
    this.minimumHeightProperty.value = orientation === Orientation.HORIZONTAL ? minimumCurrentOppositeSize : minimumAllowableSize;

    this.finishedLayoutEmitter.emit();

    lines.forEach( line => line.freeToPool() );
  }

  get justify(): HorizontalLayoutJustification | VerticalLayoutJustification {
    const result = LayoutJustification.internalToJustify( this._orientation, this._justify );

    assert && assert( LayoutJustification.getAllowedJustificationValues( this._orientation ).includes( result ) );

    return result;
  }

  set justify( value: HorizontalLayoutJustification | VerticalLayoutJustification ) {
    assert && assert( LayoutJustification.getAllowedJustificationValues( this._orientation ).includes( value ),
      `justify ${value} not supported, with the orientation ${this._orientation}, the valid values are ${LayoutJustification.getAllowedJustificationValues( this._orientation )}` );

    // remapping align values to an independent set, so they aren't orientation-dependent
    const mappedValue = LayoutJustification.justifyToInternal( this._orientation, value );

    assert && assert( mappedValue instanceof LayoutJustification );

    if ( this._justify !== mappedValue ) {
      this._justify = mappedValue;

      this.updateLayoutAutomatically();
    }
  }

  get justifyLines(): HorizontalLayoutJustification | VerticalLayoutJustification | null {
    if ( this._justifyLines === null ) {
      return null;
    }
    else {
      const result = LayoutJustification.internalToJustify( this._orientation, this._justifyLines );

      assert && assert( LayoutJustification.getAllowedJustificationValues( this._orientation ).includes( result ) );

      return result;
    }
  }

  set justifyLines( value: HorizontalLayoutJustification | VerticalLayoutJustification | null ) {
    assert && assert( value === null || LayoutJustification.getAllowedJustificationValues( this._orientation.opposite ).includes( value ),
      `justify ${value} not supported, with the orientation ${this._orientation.opposite}, the valid values are ${LayoutJustification.getAllowedJustificationValues( this._orientation.opposite )} or null` );

    // remapping align values to an independent set, so they aren't orientation-dependent
    const mappedValue = value === null ? null : LayoutJustification.justifyToInternal( this._orientation.opposite, value );

    assert && assert( mappedValue === null || mappedValue instanceof LayoutJustification );

    if ( this._justifyLines !== mappedValue ) {
      this._justifyLines = mappedValue;

      this.updateLayoutAutomatically();
    }
  }

  get wrap(): boolean {
    return this._wrap;
  }

  set wrap( value: boolean ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._wrap !== value ) {
      this._wrap = value;

      this.updateLayoutAutomatically();
    }
  }

  get spacing(): number {
    return this._spacing;
  }

  set spacing( value: number ) {
    assert && assert( typeof value === 'number' && isFinite( value ) && value >= 0 );

    if ( this._spacing !== value ) {
      this._spacing = value;

      this.updateLayoutAutomatically();
    }
  }

  get lineSpacing(): number {
    return this._lineSpacing;
  }

  set lineSpacing( value: number ) {
    assert && assert( typeof value === 'number' && isFinite( value ) && value >= 0 );

    if ( this._lineSpacing !== value ) {
      this._lineSpacing = value;

      this.updateLayoutAutomatically();
    }
  }

  insertCell( index: number, cell: FlowCell ): void {
    assert && assert( typeof index === 'number' );
    assert && assert( index >= 0 );
    assert && assert( index <= this.cells.length );
    assert && assert( cell instanceof FlowCell );
    assert && assert( !_.includes( this.cells, cell ) );

    cell.orientation = this.orientation;

    this.cells.splice( index, 0, cell );
    this.addNode( cell.node );
    cell.changedEmitter.addListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  removeCell( cell: FlowCell ): void {
    assert && assert( cell instanceof FlowCell );
    assert && assert( _.includes( this.cells, cell ) );

    arrayRemove( this.cells, cell );
    this.removeNode( cell.node );
    cell.changedEmitter.removeListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  reorderCells( cells: FlowCell[], minChangeIndex: number, maxChangeIndex: number ): void {
    this.cells.splice( minChangeIndex, maxChangeIndex - minChangeIndex + 1, ...cells );

    this.updateLayoutAutomatically();
  }

  getPreferredProperty( orientation: Orientation ): IProperty<number | null> {
    return orientation === Orientation.HORIZONTAL ? this.preferredWidthProperty : this.preferredHeightProperty;
  }

  /**
   * Releases references
   */
  override dispose(): void {
    this.cells.forEach( cell => this.removeCell( cell ) );

    super.dispose();
  }

  static create( ancestorNode: Node, options?: FlowConstraintOptions ): FlowConstraint {
    return new FlowConstraint( ancestorNode, options );
  }
}

scenery.register( 'FlowConstraint', FlowConstraint );
export { FLOW_CONSTRAINT_OPTION_KEYS };
