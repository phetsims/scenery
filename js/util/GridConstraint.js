// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import merge from '../../../phet-core/js/merge.js';
import mutate from '../../../phet-core/js/mutate.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Constraint from './Constraint.js';
import GridCell from './GridCell.js';
import GridConfigurable from './GridConfigurable.js';

const GRID_CONSTRAINT_OPTION_KEYS = [
  'excludeInvisible',
  'xSpacing',
  'ySpacing'
].concat( GridConfigurable.GRID_CONFIGURABLE_OPTION_KEYS );

// TODO: Have LayoutBox use this when we're ready
class GridConstraint extends GridConfigurable( Constraint ) {
  /**
   * @param {Node} rootNode
   * @param {Object} [options]
   */
  constructor( rootNode, options ) {
    assert && assert( rootNode instanceof Node );

    options = merge( {
      // As options, so we could hook into a Node's preferred/minimum sizes if desired
      preferredWidthProperty: new TinyProperty( null ),
      preferredHeightProperty: new TinyProperty( null ),
      minimumWidthProperty: new TinyProperty( null ),
      minimumHeightProperty: new TinyProperty( null )
    }, options );

    super( rootNode );

    // @private {Set.<GridCell>}
    this.cells = new Set();

    // @private {boolean}
    this._excludeInvisible = true;

    // @private {number|Array.<number>}
    this._xSpacing = 0;
    this._ySpacing = 0;

    // @public {Property.<Bounds2>} - Reports out the used layout bounds (may be larger than actual bounds, since it
    // will include margins, etc.)
    this.layoutBoundsProperty = new Property( Bounds2.NOTHING );

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
    mutate( this, GRID_CONSTRAINT_OPTION_KEYS, options );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );

    this.preferredWidthProperty.lazyLink( this._updateLayoutListener );
    this.preferredHeightProperty.lazyLink( this._updateLayoutListener );
  }

  /**
   * @protected
   * @override
   */
  layout() {
    super.layout();

    const preferredWidth = this.preferredWidthProperty.value;
    // const preferredHeight = this.preferredHeightProperty.value;

    const cells = [ ...this.cells ].filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
    } );

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      return;
    }

    // Handle horizontal first, so if we re-wrap we can handle vertical later.
    const columnIndices = this.getColumnIndices();
    const columnMap = new Map(); // index => column
    const columns = columnIndices.map( index => {
      const cells = this.getColumnCells( index );

      const column = {
        index: index,
        cells: cells,
        grow: _.max( cells.map( cell => cell.withDefault( 'xGrow', this ) ) ),
        min: 0,
        max: Number.POSITIVE_INFINITY,
        width: 0,
        x: 0
      };
      columnMap.set( index, column );

      return column;
    } );
    const columnSpacings = typeof this._xSpacing === 'number' ?
                           _.range( 0, columns.length - 1 ).map( () => this._xSpacing ) :
                           this._xSpacing.slice( 0, columns.length - 1 );
    assert && assert( columnSpacings.length === columns.length - 1 );
    // Scan widths for single-column cells first
    this.cells.forEach( cell => {
      if ( cell.width === 1 ) {
        const column = columnMap.get( cell.x );
        column.min = Math.max( column.min, cell.getMinimumWidth( this ) );
        column.max = Math.min( column.max, cell.getMaximumWidth( this ) );
      }
    } );
    // Then increase for spanning cells as necessary
    this.cells.forEach( cell => {
      if ( cell.width > 1 ) {
        // TODO: don't bump mins over maxes here (if columns have maxes, redistribute otherwise)
        // TODO: also handle maxes
        const columns = cell.getColumnIndices().map( index => columnMap.get( index ) );
        const currentMin = _.sum( columns.map( column => column.min ) );
        const neededMin = cell.getMinimumWidth( this );
        if ( neededMin > currentMin ) {
          const columnDelta = ( neededMin - currentMin ) / columns.length;
          columns.forEach( column => {
            column.min += columnDelta;
          } );
        }
      }
    } );
    // Adjust column widths to the min
    columns.forEach( column => {
      column.width = column.min;
    } );
    const minWidthAndSpacing = _.sum( columns.map( column => column.min ) ) + _.sum( columnSpacings );
    const width = Math.max( minWidthAndSpacing, preferredWidth || 0 );
    let widthRemaining = width - minWidthAndSpacing;
    let growableColumns;
    while ( widthRemaining > 1e-7 && ( growableColumns = columns.filter( column => {
      return column.grow > 0 && column.width < column.max - 1e-7;
    } ) ).length ) {
      const totalGrow = _.sum( growableColumns.map( column => column.grow ) );
      const amountToGrow = Math.min(
        _.min( growableColumns.map( column => ( column.max - column.width ) / column.grow ) ),
        widthRemaining / totalGrow
      );

      assert && assert( amountToGrow > 1e-11 );

      growableColumns.forEach( column => {
        column.width += amountToGrow * column.grow;
      } );
      widthRemaining -= amountToGrow * totalGrow;
    }
    // Horizontal layout
    let minX = 0;
    let maxX = width;
    columns.forEach( ( column, arrayIndex ) => {
      column.x = _.sum( columns.slice( 0, arrayIndex ).map( column => column.width ) ) + _.sum( columnSpacings.slice( 0, column.index ) );
    } );
    this.cells.forEach( cell => {
      const cellColumns = cell.getColumnIndices().map( index => columnMap.get( index ) );
      const firstColumn = columnMap.get( cell.x );
      const cellSpacings = columnSpacings.slice( cell.x, cell.x + cell.width - 1 );
      const cellAvailableWidth = _.sum( cellColumns.map( cell => cell.width ) ) + _.sum( cellSpacings );
      const cellMinimumWidth = cell.getMinimumWidth( this );
      const cellX = firstColumn.x;

      const align = cell.withDefault( 'xAlign', this );

      if ( align === GridConfigurable.Align.STRETCH ) {
        cell.attemptedPreferredWidth( this, cellAvailableWidth );
        cell.xStart( this, cellX );
      }
      else {
        cell.attemptedPreferredWidth( this, cellMinimumWidth );

        if ( align === GridConfigurable.Align.ORIGIN ) {
          // TODO: handle layout bounds
          // TODO: OMG this is horribly broken right? We would need to align stuff first
          // TODO: Do a pass to handle origin cells first (and in FLOW too)
          cell.xOrigin( this, cellX );
        }
        else {
          // TODO: optimize
          const padRatio = {
            [ GridConfigurable.Align.START ]: 0,
            [ GridConfigurable.Align.CENTER ]: 0.5,
            [ GridConfigurable.Align.END ]: 1
          }[ align ];
          cell.xStart( this, cellX + ( cellAvailableWidth - cellMinimumWidth ) * padRatio );
        }
      }

      const cellBounds = cell.getCellBounds( this );
      assert && assert( cellBounds.isFinite() );

      minX = Math.min( minX, cellBounds.minX );
      maxX = Math.max( maxX, cellBounds.maxX );
    } );

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = new Bounds2( minX, 0, maxX, 0 ); // TODO: layoutBounds
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
   * @returns {number|Array.<number>}
   */
  get xSpacing() {
    return this._xSpacing;
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set xSpacing( value ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._xSpacing !== value ) {
      this._xSpacing = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {number|Array.<number>}
   */
  get ySpacing() {
    return this._ySpacing;
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set ySpacing( value ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._ySpacing !== value ) {
      this._ySpacing = value;

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @param {GridCell} cell
   */
  addCell( cell ) {
    assert && assert( cell instanceof GridCell );
    assert && assert( !this.cells.has( cell ) );

    this.cells.add( cell );
    this.addNode( cell.node );
    cell.changedEmitter.addListener( this._updateLayoutListener );

    this.updateLayoutAutomatically();
  }

  /**
   * @public
   *
   * @param {GridCell} cell
   */
  removeCell( cell ) {
    assert && assert( cell instanceof GridCell );
    assert && assert( this.cells.has( cell ) );

    this.cells.delete( cell );
    this.removeNode( cell.node );
    cell.changedEmitter.removeListener( this._updateLayoutListener );

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

    [ ...this.cells ].forEach( cell => this.removeCell( cell ) );

    super.dispose();
  }

  /**
   * @public
   *
   * @returns {Array.<number>}
   */
  getRowIndices() {
    const result = [];

    this.cells.forEach( cell => {
      result.push( ...cell.getRowIndices() );
    } );

    return _.sortedUniq( _.sortBy( result ) );
  }

  /**
   * @public
   *
   * @returns {Array.<number>}
   */
  getColumnIndices() {
    const result = [];

    this.cells.forEach( cell => {
      result.push( ...cell.getColumnIndices() );
    } );

    return _.sortedUniq( _.sortBy( result ) );
  }

  /**
   * @public
   *
   * @param {number} row
   * @param {number} column
   * @returns {GridCell|null}
   */
  getCell( row, column ) {
    // TODO: If we have to do ridiculousness like this, just go back to array?
    return _.find( [ ...this.cells ], cell => cell.containsRow( row ) && cell.containsColumn( column ) ) || null;
  }

  /**
   * @public
   *
   * @param {number} row
   * @returns {Array.<GridCell>}
   */
  getRowCells( row ) {
    return _.filter( [ ...this.cells ], cell => cell.containsRow( row ) );
  }

  /**
   * @public
   *
   * @param {number} column
   * @returns {Array.<GridCell>}
   */
  getColumnCells( column ) {
    return _.filter( [ ...this.cells ], cell => cell.containsColumn( column ) );
  }

  /**
   * @public
   *
   * @param {Node} rootNode
   * @param {Object} [options]
   * @returns {GridConstraint}
   */
  static create( rootNode, options ) {
    return new GridConstraint( rootNode, options );
  }
}

// @public {Array.<string>}
GridConstraint.GRID_CONSTRAINT_OPTION_KEYS = GRID_CONSTRAINT_OPTION_KEYS;

scenery.register( 'GridConstraint', GridConstraint );
export default GridConstraint;