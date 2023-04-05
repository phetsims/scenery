// Copyright 2021-2023, University of Colorado Boulder

/**
 * Main flow-layout logic. Usually used indirectly through FlowBox, but can also be used directly (say, if nodes don't
 * have the same parent, or a FlowBox can't be used).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../../dot/js/Bounds2.js';
import Orientation from '../../../../phet-core/js/Orientation.js';
import arrayRemove from '../../../../phet-core/js/arrayRemove.js';
import mutate from '../../../../phet-core/js/mutate.js';
import { ExternalFlowConfigurableOptions, FLOW_CONFIGURABLE_OPTION_KEYS, FlowCell, FlowConfigurable, FlowLine, HorizontalLayoutJustification, LayoutAlign, LayoutJustification, Node, NodeLayoutAvailableConstraintOptions, NodeLayoutConstraint, scenery, VerticalLayoutJustification } from '../../imports.js';
import TProperty from '../../../../axon/js/TProperty.js';

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

  // The default spacing in-between lines long the secondary axis.
  lineSpacing?: number;

  // How extra space along the primary axis is allocated. The default is spaceBetween.
  justify?: HorizontalLayoutJustification | VerticalLayoutJustification;

  // How extra space along the secondary axis is allocated. The default is null (which will expand content to fit)
  justifyLines?: HorizontalLayoutJustification | VerticalLayoutJustification | null;

  // Whether line-wrapping is enabled. If so, the primary preferred axis will determine where things are wrapped.
  wrap?: boolean;

  // The preferred width/height (ideally from a container's localPreferredWidth/localPreferredHeight.
  preferredWidthProperty?: TProperty<number | null>;
  preferredHeightProperty?: TProperty<number | null>;

  // The minimum width/height (ideally from a container's localMinimumWidth/localMinimumHeight.
  minimumWidthProperty?: TProperty<number | null>;
  minimumHeightProperty?: TProperty<number | null>;
};
type ParentOptions = ExternalFlowConfigurableOptions & NodeLayoutAvailableConstraintOptions;
export type FlowConstraintOptions = SelfOptions & ParentOptions;

export default class FlowConstraint extends FlowConfigurable( NodeLayoutConstraint ) {

  private readonly cells: FlowCell[] = [];
  private _justify: LayoutJustification = LayoutJustification.SPACE_BETWEEN;
  private _justifyLines: LayoutJustification | null = null;
  private _wrap = false;
  private _spacing = 0;
  private _lineSpacing = 0;

  // (scenery-internal)
  public displayedCells: FlowCell[] = [];

  public constructor( ancestorNode: Node, providedOptions?: FlowConstraintOptions ) {
    super( ancestorNode, providedOptions );

    // Set configuration to actual default values (instead of null) so that we will have guaranteed non-null
    // (non-inherit) values for our computations.
    this.setConfigToBaseDefault();
    this.mutateConfigurable( providedOptions );
    mutate( this, FLOW_CONSTRAINT_OPTION_KEYS, providedOptions );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );

    this.orientationChangedEmitter.addListener( () => this.cells.forEach( cell => {
      cell.orientation = this.orientation;
    } ) );
  }

  private updateSeparatorVisibility(): void {
    // Find the index of the first visible non-separator cell. Then hide all separators until this index.
    // This is needed, so that we do NOT temporarily change the visibility of separators back-and-forth during the
    // layout. If we did that, it would trigger a layout inside every layout, leading to an infinite loop.
    // This is effectively done so that we have NO visible separators in front of the first visible non-separator cell
    // (thus satisfying our separator constraints).
    let firstVisibleNonSeparatorIndex = 0;
    for ( ; firstVisibleNonSeparatorIndex < this.cells.length; firstVisibleNonSeparatorIndex++ ) {
      const cell = this.cells[ firstVisibleNonSeparatorIndex ];
      if ( cell._isSeparator ) {
        cell.node.visible = false;
      }
      else if ( cell.node.visible ) {
        break;
      }
    }

    // Scan for separators, toggling visibility as desired. Leave the "last" separator visible, as if they are marking
    // sections "after" themselves.
    let hasVisibleNonSeparator = false;
    for ( let i = this.cells.length - 1; i > firstVisibleNonSeparatorIndex; i-- ) {
      const cell = this.cells[ i ];
      if ( cell._isSeparator ) {
        cell.node.visible = hasVisibleNonSeparator;
        hasVisibleNonSeparator = false;
      }
      else if ( cell.node.visible ) {
        hasVisibleNonSeparator = true;
      }
    }
  }

  protected override layout(): void {
    super.layout();

    // The orientation along the laid-out lines - also known as the "primary" axis
    const orientation = this._orientation;

    // The perpendicular orientation, where alignment is handled - also known as the "secondary" axis
    const oppositeOrientation = this._orientation.opposite;

    this.updateSeparatorVisibility();

    // Filter to only cells used in the layout
    const cells = this.filterLayoutCells( this.cells );

    this.displayedCells = cells;

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      this.minimumWidthProperty.value = null;
      this.minimumHeightProperty.value = null;
      return;
    }

    // Determine our preferred sizes (they can be null, in which case)
    let preferredSize: number | null = this.getPreferredProperty( orientation ).value;
    const preferredOppositeSize: number | null = this.getPreferredProperty( oppositeOrientation ).value;

    // What is the largest of the minimum sizes of cells (e.g. if we're wrapping, this would be our minimum size)
    const maxMinimumCellSize: number = Math.max( ...cells.map( cell => cell.getMinimumSize( orientation ) || 0 ) );

    // If we can't fit the content... just pretend like we have a larger preferred size!
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

        // If we're the very first cell, don't create a new line
        if ( currentLineCells.length === 0 ) {
          currentLineCells.push( cell );
          availableSpace -= cellSpace;
        }
        // Our cell fits! Epsilon for avoiding floating point issues
        else if ( this.spacing + cellSpace <= availableSpace + 1e-7 ) {
          currentLineCells.push( cell );
          availableSpace -= this.spacing + cellSpace;
        }
        // We don't fit, create a new line
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

    // Determine line opposite-orientation min/max sizes and origin sizes (how tall will a row have to be?)
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

      // If we have align:origin content, we need to see if the maximum origin span is larger than or line's
      // minimum size.
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

    // Our layout origin (usually the upper-left of the content in local coordinates, but could be different based on
    // align:origin content.
    const originPrimary = this.layoutOriginProperty.value[ orientation.coordinate ];
    const originSecondary = this.layoutOriginProperty.value[ orientation.opposite.coordinate ];

    // Primary-direction layout
    lines.forEach( line => {
      const minimumContent = _.sum( line.cells.map( cell => cell.getMinimumSize( orientation ) ) );
      const spacingAmount = this.spacing * ( line.cells.length - 1 );
      let spaceRemaining = size - minimumContent - spacingAmount;

      // Initial pending sizes
      line.cells.forEach( cell => {
        cell.size = cell.getMinimumSize( orientation );
      } );

      // Grow potential sizes if possible
      let growableCells;
      while ( spaceRemaining > 1e-7 && ( growableCells = line.cells.filter( cell => {
        // Can the cell grow more?
        return cell.effectiveGrow !== 0 && cell.size < cell.getMaximumSize( orientation ) - 1e-7;
      } ) ).length ) {
        // Total sum of "grow" values in cells that could potentially grow
        const totalGrow = _.sum( growableCells.map( cell => cell.effectiveGrow ) );
        const amountToGrow = Math.min(
          // Smallest amount that any of the cells couldn't grow past (note: proportional to effectiveGrow)
          Math.min( ...growableCells.map( cell => ( cell.getMaximumSize( orientation ) - cell.size ) / cell.effectiveGrow ) ),

          // Amount each cell grows if all of our extra space fits in ALL the cells
          spaceRemaining / totalGrow
        );

        assert && assert( amountToGrow > 1e-11 );

        growableCells.forEach( cell => {
          cell.size += amountToGrow * cell.effectiveGrow;
        } );
        spaceRemaining -= amountToGrow * totalGrow;
      }

      // Update preferred dimension based on the pending size
      line.cells.forEach( cell => cell.attemptPreferredSize( orientation, cell.size ) );

      // Gives additional spacing based on justification
      const primarySpacingFunction = this._justify.spacingFunctionFactory( spaceRemaining, line.cells.length );

      let position = originPrimary;
      line.cells.forEach( ( cell, index ) => {
        // Always include justify spacing
        position += primarySpacingFunction( index );

        // Only include normal spacing between items
        if ( index > 0 ) {
          position += this.spacing;
        }

        // ACTUALLY position it!
        cell.positionStart( orientation, position );
        cell.lastAvailableBounds[ orientation.minCoordinate ] = position;
        cell.lastAvailableBounds[ orientation.maxCoordinate ] = position + cell.size;

        position += cell.size;
        assert && assert( this.spacing >= 0 || cell.size >= -this.spacing - 1e-7,
          'Negative spacing more than a cell\'s size causes issues with layout' );
      } );
    } );

    // Secondary-direction layout
    const oppositeSpaceRemaining = oppositeSize - minimumCurrentOppositeSize;
    const initialOppositePosition = ( lines[ 0 ].hasOrigin() ? lines[ 0 ].minOrigin : 0 ) + originSecondary;
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

    // Determine the size we actually take up (localBounds for the FlowBox will use this)
    const minCoordinate = originPrimary;
    const maxCoordinate = originPrimary + size;
    const minOppositeCoordinate = initialOppositePosition;
    const maxOppositeCoordinate = initialOppositePosition + oppositeSize;

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = Bounds2.oriented(
      orientation,
      minCoordinate,
      minOppositeCoordinate,
      maxCoordinate,
      maxOppositeCoordinate
    );

    // Tell others about our new "minimum" sizes
    this.minimumWidthProperty.value = orientation === Orientation.HORIZONTAL ? minimumAllowableSize : minimumCurrentOppositeSize;
    this.minimumHeightProperty.value = orientation === Orientation.HORIZONTAL ? minimumCurrentOppositeSize : minimumAllowableSize;

    this.finishedLayoutEmitter.emit();

    lines.forEach( line => line.clean() );
  }

  public get justify(): HorizontalLayoutJustification | VerticalLayoutJustification {
    const result = LayoutJustification.internalToJustify( this._orientation, this._justify );

    assert && assert( LayoutJustification.getAllowedJustificationValues( this._orientation ).includes( result ) );

    return result;
  }

  public set justify( value: HorizontalLayoutJustification | VerticalLayoutJustification ) {
    assert && assert( LayoutJustification.getAllowedJustificationValues( this._orientation ).includes( value ),
      `justify ${value} not supported, with the orientation ${this._orientation}, the valid values are ${LayoutJustification.getAllowedJustificationValues( this._orientation )}` );

    // remapping align values to an independent set, so they aren't orientation-dependent
    const mappedValue = LayoutJustification.justifyToInternal( this._orientation, value );

    if ( this._justify !== mappedValue ) {
      this._justify = mappedValue;

      this.updateLayoutAutomatically();
    }
  }

  public get justifyLines(): HorizontalLayoutJustification | VerticalLayoutJustification | null {
    if ( this._justifyLines === null ) {
      return null;
    }
    else {
      const result = LayoutJustification.internalToJustify( this._orientation, this._justifyLines );

      assert && assert( LayoutJustification.getAllowedJustificationValues( this._orientation ).includes( result ) );

      return result;
    }
  }

  public set justifyLines( value: HorizontalLayoutJustification | VerticalLayoutJustification | null ) {
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

  public get wrap(): boolean {
    return this._wrap;
  }

  public set wrap( value: boolean ) {
    if ( this._wrap !== value ) {
      this._wrap = value;

      this.updateLayoutAutomatically();
    }
  }

  public get spacing(): number {
    return this._spacing;
  }

  public set spacing( value: number ) {
    assert && assert( isFinite( value ) );

    if ( this._spacing !== value ) {
      this._spacing = value;

      this.updateLayoutAutomatically();
    }
  }

  public get lineSpacing(): number {
    return this._lineSpacing;
  }

  public set lineSpacing( value: number ) {
    assert && assert( isFinite( value ) );

    if ( this._lineSpacing !== value ) {
      this._lineSpacing = value;

      this.updateLayoutAutomatically();
    }
  }

  public insertCell( index: number, cell: FlowCell ): void {
    assert && assert( index >= 0 );
    assert && assert( index <= this.cells.length );
    assert && assert( !_.includes( this.cells, cell ) );

    cell.orientation = this.orientation;

    this.cells.splice( index, 0, cell );
    this.addNode( cell.node );
    cell.changedEmitter.addListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  public removeCell( cell: FlowCell ): void {
    assert && assert( _.includes( this.cells, cell ) );

    arrayRemove( this.cells, cell );
    this.removeNode( cell.node );
    cell.changedEmitter.removeListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  public reorderCells( cells: FlowCell[], minChangeIndex: number, maxChangeIndex: number ): void {
    this.cells.splice( minChangeIndex, maxChangeIndex - minChangeIndex + 1, ...cells );

    this.updateLayoutAutomatically();
  }

  // (scenery-internal)
  public getPreferredProperty( orientation: Orientation ): TProperty<number | null> {
    return orientation === Orientation.HORIZONTAL ? this.preferredWidthProperty : this.preferredHeightProperty;
  }

  /**
   * Releases references
   */
  public override dispose(): void {
    // Lock during disposal to avoid layout calls
    this.lock();

    this.cells.forEach( cell => this.removeCell( cell ) );
    this.displayedCells = [];

    super.dispose();

    this.unlock();
  }

  public static create( ancestorNode: Node, options?: FlowConstraintOptions ): FlowConstraint {
    return new FlowConstraint( ancestorNode, options );
  }
}

scenery.register( 'FlowConstraint', FlowConstraint );
export { FLOW_CONSTRAINT_OPTION_KEYS };
