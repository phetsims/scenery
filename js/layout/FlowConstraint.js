// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import merge from '../../../phet-core/js/merge.js';
import mutate from '../../../phet-core/js/mutate.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Divider from './Divider.js';
import FlowCell from './FlowCell.js';
import FlowConfigurable from './FlowConfigurable.js';
import LayoutConstraint from './LayoutConstraint.js';

const FLOW_CONSTRAINT_OPTION_KEYS = [
  'spacing',
  'lineSpacing',
  'justify',
  'wrap',
  'excludeInvisible'
].concat( FlowConfigurable.FLOW_CONFIGURABLE_OPTION_KEYS );

// TODO: Have LayoutBox use this when we're ready
class FlowConstraint extends FlowConfigurable( LayoutConstraint ) {
  /**
   * @param {Node} ancestorNode
   * @param {Object} [options]
   */
  constructor( ancestorNode, options ) {
    assert && assert( ancestorNode instanceof Node );

    options = merge( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty( null ),
      preferredHeightProperty: new TinyProperty( null ),
      minimumWidthProperty: new TinyProperty( null ),
      minimumHeightProperty: new TinyProperty( null )
    }, options );

    super( ancestorNode );

    // @private {Array.<FlowCell>}
    this.cells = [];

    // @private {FlowConstraint.Justify}
    this._justify = FlowConstraint.Justify.SPACE_BETWEEN; // TODO: decide on a good default here

    // @private {boolean}
    this._wrap = false;

    // @private {number}
    this._spacing = 0;
    this._lineSpacing = 0;

    // @private {boolean}
    this._excludeInvisible = true;

    // @public {Property.<Bounds2>} - Reports out the used layout bounds (may be larger than actual bounds, since it
    // will include margins, etc.)
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

    // TODO: optimize?
    this.orientationChangedEmitter.addListener( () => {
      this.cells.forEach( cell => {
        cell.orientation = this.orientation;
      } );
    } );

    this.preferredWidthProperty.lazyLink( this._updateLayoutListener );
    this.preferredHeightProperty.lazyLink( this._updateLayoutListener );
  }

  /**
   * @protected
   * @override
   */
  layout() {
    super.layout();

    // The orientation along the laid-out lines
    const orientation = this.orientation;

    // The perpendicular orientation, where alignment is handled
    const oppositeOrientation = this.orientation.opposite;

    assert && assert( _.every( this.cells, cell => !cell.node.isDisposed ) );

    // Scan for dividers, toggling visibility as desired. Leave the "last" divider visible, as if they are marking
    // sections "after" themselves.
    let hasVisibleNonDivider = false;
    for ( let i = this.cells.length - 1; i >= 0; i-- ) {
      const cell = this.cells[ i ];
      if ( cell.node instanceof Divider ) {
        cell.node.visible = hasVisibleNonDivider;
        hasVisibleNonDivider = false;
      }
      else if ( cell.node.visible ) {
        hasVisibleNonDivider = true;
      }
    }
    // Then scan from the front, until we hit the first visible non-divider
    for ( let i = 0; i < this.cells.length; i++ ) {
      const cell = this.cells[ i ];
      if ( cell.node instanceof Divider ) {
        cell.node.visible = false;
      }
      else if ( cell.node.visible ) {
        break;
      }
    }

    const cells = this.cells.filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
    } );

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      return;
    }

    // {number|null} Determine our preferred sizes (they can be null, in which case)
    const preferredSize = orientation === Orientation.HORIZONTAL ? this.preferredWidthProperty.value : this.preferredHeightProperty.value;
    const preferredOppositeSize = orientation === Orientation.HORIZONTAL ? this.preferredHeightProperty.value : this.preferredWidthProperty.value;

    // What is the largest of the minimum sizes of cells (e.g. if we're wrapping, this would be our minimum size)
    const maxMinimumCellSize = _.max( cells.map( cell => cell.getMinimumSize( orientation, this ) ) );

    assert && assert( maxMinimumCellSize <= preferredSize || Number.POSITIVE_INFINITY, 'Will not be able to fit in this preferred size' );

    // Wrapping all of the cells into lines
    const lines = [];
    if ( this.wrap ) {
      let currentLine = [];
      let availableSpace = preferredSize || Number.POSITIVE_INFINITY;

      while ( cells.length ) {
        const cell = cells.shift();
        const cellSpace = cell.getMinimumSize( orientation, this );

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

    // {number} - Given our wrapped lines, what is our minimum size we could take up?
    const minimumCurrentSize = _.max( lines.map( line => {
      return ( line.length - 1 ) * this.spacing + _.sum( line.map( cell => cell.getMinimumSize( orientation, this ) ) );
    } ) );
    const minimumCurrentOppositeSize = _.sum( lines.map( line => {
      return _.max( line.map( cell => cell.getMinimumSize( oppositeOrientation, this ) ) );
    } ) ) + ( lines.length - 1 ) * this.lineSpacing;

    // Used for determining our "minimum" size for preferred sizes... if wrapping is enabled, we can be smaller than
    // current minimums
    const minimumAllowableSize = this.wrap ? maxMinimumCellSize : minimumCurrentSize;

    // Tell others about our new "minimum" sizes
    this.minimumWidthProperty.value = orientation === Orientation.HORIZONTAL ? minimumAllowableSize : minimumCurrentOppositeSize;
    this.minimumHeightProperty.value = orientation === Orientation.HORIZONTAL ? minimumCurrentOppositeSize : minimumAllowableSize;

    // {number} - Increase things if our preferred size is larger than our minimums (we'll figure out how to compensate
    // for the extra space below).
    const size = Math.max( minimumCurrentSize, preferredSize || 0 );

    const minCoordinate = 0;
    const maxCoordinate = size;
    let minOppositeCoordinate = Number.POSITIVE_INFINITY;
    let maxOppositeCoordinate = Number.NEGATIVE_INFINITY;

    // Primary-direction layout
    lines.forEach( line => {
      const minimumContent = _.sum( line.map( cell => cell.getMinimumSize( orientation, this ) ) );
      const spacingAmount = this.spacing * ( line.length - 1 );
      let spaceRemaining = size - minimumContent - spacingAmount;

      // Initial pending sizes
      line.forEach( cell => {
        cell._pendingSize = cell.getMinimumSize( orientation, this );
      } );

      // Grow potential sizes if possible
      // TODO: This looks unfun to read... check this with a fresh mind
      let growableCells;
      while ( spaceRemaining > 1e-7 && ( growableCells = line.filter( cell => {
        const grow = cell.withDefault( 'grow', this );
        if ( grow === 0 ) {
          return false;
        }
        return cell._pendingSize < cell.getMaximumSize( orientation, this ) - 1e-7;
      } ) ).length ) {
        const totalGrow = _.sum( growableCells.map( cell => cell.grow ) );
        const amountToGrow = Math.min(
          _.min( growableCells.map( cell => ( cell.getMaximumSize( orientation, this ) - cell._pendingSize ) / cell.grow ) ),
          spaceRemaining / totalGrow
        );

        assert && assert( amountToGrow > 1e-11 );

        growableCells.forEach( cell => {
          cell._pendingSize += amountToGrow * cell.grow;
        } );
        spaceRemaining -= amountToGrow * totalGrow;
      }

      // Update preferred dimension based on the pending size
      line.forEach( cell => cell.attemptPreferredSize( orientation, this, cell._pendingSize ) );

      // TODO: optimize, OMG, but is this generally a good idea?
      // TODO: Only I would write this code in this mental state?
      const spacingFunction = {
        [ FlowConstraint.Justify.START ]: index => 0,
        [ FlowConstraint.Justify.END ]: index => index === 0 ? spaceRemaining : 0,
        [ FlowConstraint.Justify.CENTER ]: index => index === 0 ? spaceRemaining / 2 : 0,
        [ FlowConstraint.Justify.SPACE_BETWEEN ]: index => index !== 0 ? ( spaceRemaining / ( line.length - 1 ) ) : 0,
        [ FlowConstraint.Justify.SPACE_AROUND ]: index => ( index !== 0 ? 2 : 1 ) * spaceRemaining / ( 2 * line.length ),
        [ FlowConstraint.Justify.SPACE_EVENLY ]: index => spaceRemaining / ( line.length + 1 )
      }[ this._justify ];

      let position = 0;

      line.forEach( ( cell, index ) => {
        position += spacingFunction( index );
        if ( index > 0 ) {
          position += this.spacing;
        }
        // TODO: handle coordinate transforms properly
        cell.positionStart( orientation, this, position );
        position += cell._pendingSize;
      } );
    } );

    // Secondary-direction layout
    let secondaryPosition = 0;
    lines.forEach( line => {
      const maximumSize = _.max( line.map( cell => cell.getMinimumSize( oppositeOrientation, this ) ) );

      line.forEach( cell => {
        const align = cell.withDefault( 'align', this );
        const size = cell.getMinimumSize( oppositeOrientation, this );

        if ( align === FlowConfigurable.Align.STRETCH ) {
          cell.attemptPreferredSize( oppositeOrientation, this, maximumSize );
          cell.positionStart( oppositeOrientation, this, secondaryPosition );
        }
        else {
          cell.attemptPreferredSize( oppositeOrientation, this, size );

          if ( align === FlowConfigurable.Align.ORIGIN ) {
            // TODO: handle layout bounds
            cell.positionOrigin( oppositeOrientation, this, secondaryPosition );
          }
          else {
            // TODO: optimize
            const padRatio = {
              [ FlowConfigurable.Align.START ]: 0,
              [ FlowConfigurable.Align.CENTER ]: 0.5,
              [ FlowConfigurable.Align.END ]: 1
            }[ align ];
            cell.positionStart( oppositeOrientation, this, secondaryPosition + ( maximumSize - size ) * padRatio );
          }
        }

        const cellBounds = cell.getCellBounds( this );
        assert && assert( cellBounds.isFinite() );

        minOppositeCoordinate = Math.min( minOppositeCoordinate, oppositeOrientation === Orientation.HORIZONTAL ? cellBounds.minX : cellBounds.minY );
        maxOppositeCoordinate = Math.max( maxOppositeCoordinate, oppositeOrientation === Orientation.HORIZONTAL ? cellBounds.maxX : cellBounds.maxY );
      } );

      // TODO: This is insufficient for origin, if we wrap, our origin setup will be off
      secondaryPosition += maximumSize + this.lineSpacing;
    } );

    // TODO: align-content flexbox equivalent
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

    this.finishedLayoutEmitter.emit();
  }

  /**
   * @public
   *
   * @returns {FlowConstraint.Justify}
   */
  get justify() {
    return this._justify;
  }

  /**
   * @public
   *
   * @param {FlowConstraint.Justify|string} value
   */
  set justify( value ) {
    if ( typeof value === 'string' ) {
      value = justifyMap[ value ];
    }

    assert && assert( FlowConstraint.Justify.includes( value ) );

    if ( this._justify !== value ) {
      this._justify = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get wrap() {
    return this._wrap;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set wrap( value ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._wrap !== value ) {
      this._wrap = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get spacing() {
    return this._spacing;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set spacing( value ) {
    assert && assert( typeof value === 'number' && isFinite( value ) && value >= 0 );

    if ( this._spacing !== value ) {
      this._spacing = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get lineSpacing() {
    return this._lineSpacing;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set lineSpacing( value ) {
    assert && assert( typeof value === 'number' && isFinite( value ) && value >= 0 );

    if ( this._lineSpacing !== value ) {
      this._lineSpacing = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {string}
   */
  get excludeInvisible() {
    return this._excludeInvisible;
  }

  /**
   * @public
   *
   * @param {Orientation|string} value
   */
  set excludeInvisible( value ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._excludeInvisible !== value ) {
      this._excludeInvisible = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @param {number} index
   * @param {FlowCell} cell
   */
  insertCell( index, cell ) {
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

  /**
   * @public
   *
   * @param {FlowCell} cell
   */
  removeCell( cell ) {
    assert && assert( cell instanceof FlowCell );
    assert && assert( _.includes( this.cells, cell ) );

    arrayRemove( this.cells, cell );
    this.removeNode( cell.node );
    cell.changedEmitter.removeListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  /**
   * @public
   *
   * @param {Array.<FlowCell>} cells
   * @param {number} minChangeIndex
   * @param {number} maxChangeIndex
   */
  reorderCells( cells, minChangeIndex, maxChangeIndex ) {
    // TODO: assertions for this!!! So many things could go wrong here

    this.cells.splice( minChangeIndex, maxChangeIndex - minChangeIndex + 1, ...cells );

    this.updateLayoutAutomatically();
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    // In case they're from external sources
    this.preferredWidthProperty.unlink( this._updateLayoutListener );
    this.preferredHeightProperty.unlink( this._updateLayoutListener );

    this.cells.forEach( cell => this.removeCell( cell ) );

    super.dispose();
  }

  /**
   * @public
   *
   * @param {Node} ancestorNode
   * @param {Object} [options]
   * @returns {FlowConstraint}
   */
  static create( ancestorNode, options ) {
    return new FlowConstraint( ancestorNode, options );
  }
}

FlowConstraint.Justify = Enumeration.byKeys( [
  'START',
  'END',
  'CENTER',
  'SPACE_BETWEEN',
  'SPACE_AROUND',
  'SPACE_EVENLY'
] );
const justifyMap = {
  start: FlowConstraint.Justify.START,
  top: FlowConstraint.Justify.START,
  left: FlowConstraint.Justify.START,

  end: FlowConstraint.Justify.END,
  bottom: FlowConstraint.Justify.END,
  right: FlowConstraint.Justify.END,

  center: FlowConstraint.Justify.CENTER,
  spaceBetween: FlowConstraint.Justify.SPACE_BETWEEN,
  spaceAround: FlowConstraint.Justify.SPACE_AROUND,
  spaceEvenly: FlowConstraint.Justify.SPACE_EVENLY
};

// @public {Array.<string>}
FlowConstraint.FLOW_CONSTRAINT_OPTION_KEYS = FLOW_CONSTRAINT_OPTION_KEYS;

scenery.register( 'FlowConstraint', FlowConstraint );
export default FlowConstraint;