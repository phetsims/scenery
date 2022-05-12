// Copyright 2021-2022, University of Colorado Boulder

/**
 * Main flow-layout logic. Usually used indirectly through FlowBox, but can also be used directly (say, if nodes don't
 * have the same parent, or a FlowBox can't be used).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import optionize from '../../../phet-core/js/optionize.js';
import mutate from '../../../phet-core/js/mutate.js';
import { scenery, Node, Divider, FlowCell, FlowConfigurable, LayoutConstraint, FLOW_CONFIGURABLE_OPTION_KEYS, FlowConfigurableOptions, FlowConfigurableAlign } from '../imports.js';
import EnumerationValue from '../../../phet-core/js/EnumerationValue.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import IProperty from '../../../axon/js/IProperty.js';

const FLOW_CONSTRAINT_OPTION_KEYS = [
  'spacing',
  'lineSpacing',
  'justify',
  'wrap',
  'excludeInvisible'
].concat( FLOW_CONFIGURABLE_OPTION_KEYS );

const flowHorizontalJustificationValues = [ 'left', 'right', 'center', 'spaceBetween', 'spaceAround', 'spaceEvenly' ] as const;
const flowVerticalJustificationValues = [ 'top', 'bottom', 'center', 'spaceBetween', 'spaceAround', 'spaceEvenly' ] as const;

export type FlowHorizontalJustification = typeof flowHorizontalJustificationValues[number];
export type FlowVerticalJustification = typeof flowVerticalJustificationValues[number];

const getAllowedJustificationValues = ( orientation: Orientation ): readonly string[] => {
  return orientation === Orientation.HORIZONTAL ? flowHorizontalJustificationValues : flowVerticalJustificationValues;
};

type SpaceRemainingFunctionFactory = ( spaceRemaining: number, lineLength: number ) => ( ( index: number ) => number );

class FlowConstraintJustify extends EnumerationValue {
  static readonly START = new FlowConstraintJustify(
    () => () => 0,
    'left', 'top'
  );

  static readonly END = new FlowConstraintJustify(
    spaceRemaining => index => index === 0 ? spaceRemaining : 0,
    'right', 'bottom'
  );

  static readonly CENTER = new FlowConstraintJustify(
    spaceRemaining => index => index === 0 ? spaceRemaining / 2 : 0,
    'center', 'center'
  );

  static readonly SPACE_BETWEEN = new FlowConstraintJustify(
    ( spaceRemaining, lineLength ) => index => index !== 0 ? ( spaceRemaining / ( lineLength - 1 ) ) : 0,
    'spaceBetween', 'spaceBetween'
  );

  static readonly SPACE_AROUND = new FlowConstraintJustify(
    ( spaceRemaining, lineLength ) => index => ( index !== 0 ? 2 : 1 ) * spaceRemaining / ( 2 * lineLength ),
    'spaceAround', 'spaceAround'
  );

  static readonly SPACE_EVENLY = new FlowConstraintJustify(
    ( spaceRemaining, lineLength ) => index => spaceRemaining / ( lineLength + 1 ),
    'spaceEvenly', 'spaceEvenly'
  );

  readonly horizontal: FlowHorizontalJustification;
  readonly vertical: FlowVerticalJustification;
  readonly spacingFunctionFactory: SpaceRemainingFunctionFactory;

  constructor( spacingFunctionFactory: SpaceRemainingFunctionFactory, horizontal: FlowHorizontalJustification, vertical: FlowVerticalJustification ) {
    super();

    this.spacingFunctionFactory = spacingFunctionFactory;
    this.horizontal = horizontal;
    this.vertical = vertical;
  }

  static readonly enumeration = new Enumeration( FlowConstraintJustify, {
    phetioDocumentation: 'Justify for FlowConstraint'
  } );
}

const horizontalJustifyMap = {
  left: FlowConstraintJustify.START,
  right: FlowConstraintJustify.END,
  center: FlowConstraintJustify.CENTER,
  spaceBetween: FlowConstraintJustify.SPACE_BETWEEN,
  spaceAround: FlowConstraintJustify.SPACE_AROUND,
  spaceEvenly: FlowConstraintJustify.SPACE_EVENLY
};
const verticalJustifyMap = {
  top: FlowConstraintJustify.START,
  bottom: FlowConstraintJustify.END,
  center: FlowConstraintJustify.CENTER,
  spaceBetween: FlowConstraintJustify.SPACE_BETWEEN,
  spaceAround: FlowConstraintJustify.SPACE_AROUND,
  spaceEvenly: FlowConstraintJustify.SPACE_EVENLY
};
const justifyToInternal = ( orientation: Orientation, key: FlowHorizontalJustification | FlowVerticalJustification ): FlowConstraintJustify => {
  if ( orientation === Orientation.HORIZONTAL ) {
    assert && assert( horizontalJustifyMap[ key as 'left' | 'right' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ] );

    return horizontalJustifyMap[ key as 'left' | 'right' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ];
  }
  else {
    assert && assert( verticalJustifyMap[ key as 'top' | 'bottom' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ] );

    return verticalJustifyMap[ key as 'top' | 'bottom' | 'center' | 'spaceBetween' | 'spaceAround' | 'spaceEvenly' ];
  }
};
const internalToJustify = ( orientation: Orientation, justify: FlowConstraintJustify ): FlowHorizontalJustification | FlowVerticalJustification => {
  if ( orientation === Orientation.HORIZONTAL ) {
    return justify.horizontal;
  }
  else {
    return justify.vertical;
  }
};

type SelfOptions = {
  // The default spacing in-between elements in the primary direction. If additional (or less) spacing is desired for
  // certain elements, per-element margins (even negative) can be set in the layoutOptions of nodes contained.
  spacing?: number;

  // The default spacing in-between lines in the secondary direction.
  lineSpacing?: number;

  // How extra space in the primary direction is allocated. The default is spaceBetween.
  justify?: FlowHorizontalJustification | FlowVerticalJustification;

  // Whether line-wrapping is enabled. If so, the primary preferred dimension will determine where things are wrapped.
  wrap?: boolean;

  // Whether invisible Nodes are excluded from the layout.
  excludeInvisible?: boolean;

  // The preferred width/height (ideally from a container's localPreferredWidth/localPreferredHeight.
  preferredWidthProperty?: IProperty<number | null>;
  preferredHeightProperty?: IProperty<number | null>;

  // The minimum width/height (ideally from a container's localMinimumWidth/localMinimumHeight.
  minimumWidthProperty?: IProperty<number | null>;
  minimumHeightProperty?: IProperty<number | null>;
};

export type FlowConstraintOptions = SelfOptions & FlowConfigurableOptions;

export default class FlowConstraint extends FlowConfigurable( LayoutConstraint ) {

  private readonly cells: FlowCell[];
  private _justify: FlowConstraintJustify;
  private _wrap: boolean;
  private _spacing: number;
  private _lineSpacing: number;
  private _excludeInvisible: boolean;

  readonly preferredWidthProperty: IProperty<number | null>;
  readonly preferredHeightProperty: IProperty<number | null>;
  readonly minimumWidthProperty: IProperty<number | null>;
  readonly minimumHeightProperty: IProperty<number | null>;

  // Reports out the used layout bounds (may be larger than actual bounds, since it
  // will include margins, etc.)
  readonly layoutBoundsProperty: IProperty<Bounds2>;

  constructor( ancestorNode: Node, providedOptions?: FlowConstraintOptions ) {
    assert && assert( ancestorNode instanceof Node );

    const options = optionize<FlowConstraintOptions, Pick<SelfOptions, 'preferredWidthProperty' | 'preferredHeightProperty' | 'minimumWidthProperty' | 'minimumHeightProperty'>>()( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty<number | null>( null ),
      preferredHeightProperty: new TinyProperty<number | null>( null ),
      minimumWidthProperty: new TinyProperty<number | null>( null ),
      minimumHeightProperty: new TinyProperty<number | null>( null )
    }, providedOptions );

    super( ancestorNode );

    this.cells = [];
    this._justify = FlowConstraintJustify.SPACE_BETWEEN;
    this._wrap = false;
    this._spacing = 0;
    this._lineSpacing = 0;
    this._excludeInvisible = true;

    this.layoutBoundsProperty = new Property( Bounds2.NOTHING, {
      useDeepEquality: true
    } );

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
    mutate( this, FLOW_CONSTRAINT_OPTION_KEYS, options );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );

    this.orientationChangedEmitter.addListener( () => this.cells.forEach( cell => {
      cell.orientation = this.orientation;
    } ) );

    this.preferredWidthProperty.lazyLink( this._updateLayoutListener );
    this.preferredHeightProperty.lazyLink( this._updateLayoutListener );
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
      return cell.isConnected() && cell.proxy.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
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

    // Wrapping all of the cells into lines
    const lines = [];
    if ( this.wrap ) {
      let currentLine: FlowCell[] = [];
      let availableSpace = preferredSize || Number.POSITIVE_INFINITY;

      while ( cells.length ) {
        const cell = cells.shift()!;
        const cellSpace = cell.getMinimumSize( orientation );

        if ( currentLine.length === 0 ) {
          currentLine.push( cell );
          availableSpace -= cellSpace;
        }
        else if ( this.spacing + cellSpace <= availableSpace + 1e-7 ) {
          currentLine.push( cell );
          availableSpace -= this.spacing + cellSpace;
        }
        else {
          lines.push( currentLine );
          availableSpace = preferredSize || Number.POSITIVE_INFINITY;

          currentLine = [ cell ];
          availableSpace -= cellSpace;
        }
      }

      if ( currentLine.length ) {
        lines.push( currentLine );
      }
    }
    else {
      lines.push( cells );
    }

    // Given our wrapped lines, what is our minimum size we could take up?
    const minimumCurrentSize: number = Math.max( ...lines.map( line => {
      return ( line.length - 1 ) * this.spacing + _.sum( line.map( cell => cell.getMinimumSize( orientation ) ) );
    } ) );
    const minimumCurrentOppositeSize = _.sum( lines.map( line => {
      return _.max( line.map( cell => cell.getMinimumSize( oppositeOrientation ) ) );
    } ) ) + ( lines.length - 1 ) * this.lineSpacing;

    // Used for determining our "minimum" size for preferred sizes... if wrapping is enabled, we can be smaller than
    // current minimums
    const minimumAllowableSize = this.wrap ? maxMinimumCellSize : minimumCurrentSize;

    // Increase things if our preferred size is larger than our minimums (we'll figure out how to compensate
    // for the extra space below).
    const size: number = Math.max( minimumCurrentSize, preferredSize || 0 );

    const minCoordinate = 0;
    const maxCoordinate = size;
    let minOppositeCoordinate = Number.POSITIVE_INFINITY;
    let maxOppositeCoordinate = Number.NEGATIVE_INFINITY;

    // Primary-direction layout
    lines.forEach( line => {
      const minimumContent = _.sum( line.map( cell => cell.getMinimumSize( orientation ) ) );
      const spacingAmount = this.spacing * ( line.length - 1 );
      let spaceRemaining = size - minimumContent - spacingAmount;

      // Initial pending sizes
      line.forEach( cell => {
        cell._pendingSize = cell.getMinimumSize( orientation );
      } );

      // Grow potential sizes if possible
      let growableCells;
      while ( spaceRemaining > 1e-7 && ( growableCells = line.filter( cell => {
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
      line.forEach( cell => cell.attemptPreferredSize( orientation, cell._pendingSize ) );

      const spacingFunction = this._justify.spacingFunctionFactory( spaceRemaining, line.length );

      let position = 0;

      line.forEach( ( cell, index ) => {
        position += spacingFunction( index );
        if ( index > 0 ) {
          position += this.spacing;
        }
        cell.positionStart( orientation, position );
        position += cell._pendingSize;
      } );
    } );

    // Secondary-direction layout
    let secondaryPosition = 0;
    lines.forEach( ( line, index ) => {
      // Mimicking https://www.w3.org/TR/css-flexbox-1/#align-items-property for baseline (for our origin)
      // Origin will sync all origin-based items (so their origin matches), and then position ALL of that as if it was
      // align left or top (depending on the orientation).

      // Find the union of all origin-based cells (will be Bounds2.NOTHING if there are none)
      const maximumOriginBounds = line.reduce( ( bounds: Bounds2, cell: FlowCell ) => {
        return cell.effectiveAlign === FlowConfigurableAlign.ORIGIN ? bounds.union( cell.getOriginBounds() ) : bounds;
      }, Bounds2.NOTHING );

      // When we lay out everything with align:origin, what is the dimension of all of that content?
      const maximumOriginSize = maximumOriginBounds.isFinite() ? maximumOriginBounds[ oppositeOrientation.size ] : 0;

      // TODO: using preferredOppositeSize doesn't seem to mesh well with wrapping, no? Perhaps align-content equivalent would fix this?
      const lineSize = Math.max( preferredOppositeSize || 0, maximumOriginSize, ...line.map( cell => cell.getMinimumSize( oppositeOrientation ) || 0 ) );

      // The distance from the "start" (top/left) to the origin, for origin-aligned items
      const originOffset = maximumOriginBounds.isValid() ? -maximumOriginBounds.top : 0;

      // The position of the "start" (top/left) for the entire line
      const lineStartPosition = index === 0 ? -originOffset : secondaryPosition;

      if ( index === 0 ) {
        minOppositeCoordinate = lineStartPosition;
      }
      if ( index === lines.length - 1 ) {
        maxOppositeCoordinate = lineStartPosition + lineSize;
      }

      line.forEach( cell => {
        const align = cell.effectiveAlign;

        // If stretching, we'll expand to the lineSize (otherwise will use the minimum size available)
        const preferredSize = ( cell.effectiveStretch && cell.isSizable( oppositeOrientation ) ) ? lineSize : cell.getMinimumSize( oppositeOrientation );

        cell.attemptPreferredSize( oppositeOrientation, preferredSize );

        if ( align === FlowConfigurableAlign.ORIGIN ) {
          cell.positionOrigin( oppositeOrientation, lineStartPosition + originOffset );
        }
        else {
          cell.positionStart( oppositeOrientation, lineStartPosition + ( lineSize - cell.proxy[ oppositeOrientation.size ] ) * align.padRatio );
        }

        const cellBounds = cell.getCellBounds();
        assert && assert( cellBounds.isFinite() );
      } );

      const incrementAmount = lineSize + this.lineSpacing;

      if ( index === 0 ) {
        // If we're the first line, we need to include the lineStartPosition (if we had something origin-positioned,
        // we want to have the first line's position origin be exactly 0). We'll start from there, and then each line
        // in the future will be simpler with an increment (originOffset will handle things nicely)
        secondaryPosition = lineStartPosition + incrementAmount;
      }
      else {
        // Otherwise we can just increment by the line size (and add in spacing)
        secondaryPosition += incrementAmount;
      }
    } );

    // TODO: align-content flexbox equivalent (this is possibly not needed, since we are typically not wrapping?
    // For now, we'll just pad ourself out
    if ( preferredOppositeSize && ( maxOppositeCoordinate - minOppositeCoordinate ) < preferredOppositeSize ) {
      maxOppositeCoordinate = minOppositeCoordinate + preferredOppositeSize;
    }

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
  }

  get justify(): FlowHorizontalJustification | FlowVerticalJustification {
    const result = internalToJustify( this._orientation, this._justify );

    assert && assert( getAllowedJustificationValues( this._orientation ).includes( result ) );

    return result;
  }

  set justify( value: FlowHorizontalJustification | FlowVerticalJustification ) {
    assert && assert( getAllowedJustificationValues( this._orientation ).includes( value ),
      `justify ${value} not supported, with the orientation ${this._orientation}, the valid values are ${getAllowedJustificationValues( this._orientation )}` );

    // remapping align values to an independent set, so they aren't orientation-dependent
    const mappedValue = justifyToInternal( this._orientation, value );

    assert && assert( mappedValue instanceof FlowConstraintJustify );

    if ( this._justify !== mappedValue ) {
      this._justify = mappedValue;

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
    // In case they're from external sources
    this.preferredWidthProperty.unlink( this._updateLayoutListener );
    this.preferredHeightProperty.unlink( this._updateLayoutListener );

    this.cells.forEach( cell => this.removeCell( cell ) );

    super.dispose();
  }

  static create( ancestorNode: Node, options?: FlowConstraintOptions ): FlowConstraint {
    return new FlowConstraint( ancestorNode, options );
  }
}

scenery.register( 'FlowConstraint', FlowConstraint );
export { FLOW_CONSTRAINT_OPTION_KEYS };
