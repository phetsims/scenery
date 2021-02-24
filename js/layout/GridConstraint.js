// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Property from '../../../axon/js/Property.js';
import TinyProperty from '../../../axon/js/TinyProperty.js';
import Bounds2 from '../../../dot/js/Bounds2.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import merge from '../../../phet-core/js/merge.js';
import mutate from '../../../phet-core/js/mutate.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Constraint from './Constraint.js';
import GridCell from './GridCell.js';
import GridConfigurable from './GridConfigurable.js';

const GRID_CONSTRAINT_OPTION_KEYS = [
  'excludeInvisible',
  'spacing',
  'xSpacing',
  'ySpacing'
].concat( GridConfigurable.GRID_CONFIGURABLE_OPTION_KEYS );

// TODO: Have LayoutBox use this when we're ready
class GridConstraint extends GridConfigurable( Constraint ) {
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

    // @private {Set.<GridCell>}
    this.cells = new Set();

    // @private {boolean}
    this._excludeInvisible = true;

    // @private {OrientationPair.<number|Array.<number>>}
    this._spacing = new OrientationPair( 0, 0 );

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

    const cells = [ ...this.cells ].filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
    } );

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      return;
    }

    const preferredSizes = new OrientationPair( this.preferredWidthProperty.value, this.preferredHeightProperty.value );
    const layoutBounds = new Bounds2( 0, 0, 0, 0 ); // TODO: Bounds2.NOTHING.copy() once we have both dimensions handled

    // Handle horizontal first, so if we re-wrap we can handle vertical later.
    [ Orientation.HORIZONTAL, Orientation.VERTICAL ].forEach( orientation => {
      const orientedSpacing = this._spacing.get( orientation );
      const capLetter = orientation === Orientation.HORIZONTAL ? 'X' : 'Y';
      const minField = `min${capLetter}`;
      const maxField = `max${capLetter}`;

      const lineIndices = this.getIndices( orientation );
      const lineMap = new Map(); // index => line
      const lines = lineIndices.map( index => {
        const cells = this.getCells( orientation, index );

        // TODO: poolable type?
        const line = {
          index: index,
          cells: cells,
          grow: _.max( cells.map( cell => cell.withDefault( orientation === Orientation.HORIZONTAL ? 'xGrow' : 'yGrow', this ) ) ),
          min: 0,
          max: Number.POSITIVE_INFINITY,
          size: 0,
          position: 0
        };
        lineMap.set( index, line );

        return line;
      } );
      const lineSpacings = typeof orientedSpacing === 'number' ?
                             _.range( 0, lines.length - 1 ).map( () => orientedSpacing ) :
                             orientedSpacing.slice( 0, lines.length - 1 );
      assert && assert( lineSpacings.length === lines.length - 1 );

      // Scan sizes for single-line cells first
      this.cells.forEach( cell => {
        if ( cell.size.get( orientation ) === 1 ) {
          const line = lineMap.get( cell.position.get( orientation ) );
          line.min = Math.max( line.min, cell.getMinimumSize( orientation, this ) );
          line.max = Math.min( line.max, cell.getMaximumSize( orientation, this ) );
        }
      } );

      // Then increase for spanning cells as necessary
      this.cells.forEach( cell => {
        if ( cell.size.get( orientation ) > 1 ) {
          // TODO: don't bump mins over maxes here (if lines have maxes, redistribute otherwise)
          // TODO: also handle maxes
          const lines = cell.getIndices( orientation ).map( index => lineMap.get( index ) );
          const currentMin = _.sum( lines.map( line => line.min ) );
          const neededMin = cell.getMinimumSize( orientation, this );
          if ( neededMin > currentMin ) {
            const lineDelta = ( neededMin - currentMin ) / lines.length;
            lines.forEach( line => {
              line.min += lineDelta;
            } );
          }
        }
      } );

      // Adjust line sizes to the min
      lines.forEach( line => {
        line.size = line.min;
      } );
      const minSizeAndSpacing = _.sum( lines.map( line => line.min ) ) + _.sum( lineSpacings );
      const size = Math.max( minSizeAndSpacing, preferredSizes.get( orientation ) || 0 );
      let sizeRemaining = size - minSizeAndSpacing;
      let growableLines;
      while ( sizeRemaining > 1e-7 && ( growableLines = lines.filter( column => {
        return column.grow > 0 && column.size < column.max - 1e-7;
      } ) ).length ) {
        const totalGrow = _.sum( growableLines.map( column => column.grow ) );
        const amountToGrow = Math.min(
          _.min( growableLines.map( column => ( column.max - column.size ) / column.grow ) ),
          sizeRemaining / totalGrow
        );

        assert && assert( amountToGrow > 1e-11 );

        growableLines.forEach( column => {
          column.size += amountToGrow * column.grow;
        } );
        sizeRemaining -= amountToGrow * totalGrow;
      }

      // Layout
      layoutBounds[ minField ] = 0;
      layoutBounds[ maxField ] = size;
      lines.forEach( ( column, arrayIndex ) => {
        column.position = _.sum( lines.slice( 0, arrayIndex ).map( column => column.size ) ) + _.sum( lineSpacings.slice( 0, column.index ) );
      } );
      this.cells.forEach( cell => {
        const cellIndexPosition = cell.position.get( orientation );
        const cellSize = cell.size.get( orientation );
        const cellLines = cell.getIndices( orientation ).map( index => lineMap.get( index ) );
        const firstColumn = lineMap.get( cellIndexPosition );
        const cellSpacings = lineSpacings.slice( cellIndexPosition, cellIndexPosition + cellSize - 1 );
        const cellAvailableSize = _.sum( cellLines.map( line => line.size ) ) + _.sum( cellSpacings );
        const cellMinimumSize = cell.getMinimumSize( orientation, this );
        const cellPosition = firstColumn.position;

        const align = cell.withDefault( orientation === Orientation.HORIZONTAL ? 'xAlign' : 'yAlign', this );

        if ( align === GridConfigurable.Align.STRETCH ) {
          cell.attemptedPreferredSize( orientation, this, cellAvailableSize );
          cell.positionStart( orientation, this, cellPosition );
        }
        else {
          cell.attemptedPreferredSize( orientation, this, cellMinimumSize );

          if ( align === GridConfigurable.Align.ORIGIN ) {
            // TODO: handle layout bounds
            // TODO: OMG this is horribly broken right? We would need to align stuff first
            // TODO: Do a pass to handle origin cells first (and in FLOW too)
            cell.positionOrigin( orientation, this, cellPosition );
          }
          else {
            // TODO: optimize
            const padRatio = {
              [ GridConfigurable.Align.START ]: 0,
              [ GridConfigurable.Align.CENTER ]: 0.5,
              [ GridConfigurable.Align.END ]: 1
            }[ align ];
            cell.positionStart( orientation, this, cellPosition + ( cellAvailableSize - cellMinimumSize ) * padRatio );
          }
        }

        const cellBounds = cell.getCellBounds( this );
        assert && assert( cellBounds.isFinite() );

        layoutBounds[ minField ] = Math.min( layoutBounds[ minField ], cellBounds[ minField ] );
        layoutBounds[ maxField ] = Math.max( layoutBounds[ maxField ], cellBounds[ maxField ] );
      } );
    } );

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = layoutBounds;
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
  get spacing() {
    assert && assert( this.xSpacing === this.ySpacing );

    return this.xSpacing;
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set spacing( value ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.HORIZONTAL ) !== value || this._spacing.get( Orientation.VERTICAL ) !== value ) {
      this._spacing.set( Orientation.HORIZONTAL, value );
      this._spacing.set( Orientation.VERTICAL, value );

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {number|Array.<number>}
   */
  get xSpacing() {
    return this._spacing.get( Orientation.HORIZONTAL );
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set xSpacing( value ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.HORIZONTAL ) !== value ) {
      this._spacing.set( Orientation.HORIZONTAL, value );

      this.updateLayoutAutomatically();
    }
  }

  /**
   * @public
   *
   * @returns {number|Array.<number>}
   */
  get ySpacing() {
    return this._spacing.get( Orientation.VERTICAL );
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set ySpacing( value ) {
    assert && assert( ( typeof value === 'number' && isFinite( value ) && value >= 0 ) ||
                      ( Array.isArray( value ) && _.every( value, item => ( typeof item === 'number' && isFinite( item ) && item >= 0 ) ) ) );

    if ( this._spacing.get( Orientation.VERTICAL ) !== value ) {
      this._spacing.set( Orientation.VERTICAL, value );

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
   * @param {Orientation} orientation
   * @returns {Array.<number>}
   */
  getIndices( orientation ) {
    const result = [];

    this.cells.forEach( cell => {
      result.push( ...cell.getIndices( orientation ) );
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
   * @param {Orientation} orientation
   * @param {number} index
   */
  getCells( orientation, index ) {
    return _.filter( [ ...this.cells ], cell => cell.containsIndex( orientation, index ) );
  }

  /**
   * @public
   *
   * @param {Node} ancestorNode
   * @param {Object} [options]
   * @returns {GridConstraint}
   */
  static create( ancestorNode, options ) {
    return new GridConstraint( ancestorNode, options );
  }
}

// @public {Array.<string>}
GridConstraint.GRID_CONSTRAINT_OPTION_KEYS = GRID_CONSTRAINT_OPTION_KEYS;

scenery.register( 'GridConstraint', GridConstraint );
export default GridConstraint;