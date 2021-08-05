// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import GridConfigurable from './GridConfigurable.js';

// {Object.<GridConfigurable.Align,number>}
const padRatioMap = {
  [ GridConfigurable.Align.STRETCH ]: 0,
  [ GridConfigurable.Align.START ]: 0,
  [ GridConfigurable.Align.CENTER ]: 0.5,
  [ GridConfigurable.Align.END ]: 1
};

const sizableFlagPair = new OrientationPair( 'widthSizable', 'heightSizable' );
const minimumSizePair = new OrientationPair( 'minimumWidth', 'minimumHeight' );
const preferredSizePair = new OrientationPair( 'preferredWidth', 'preferredHeight' );

// {number} - Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

class GridCell extends GridConfigurable( Object ) {
  /**
   * @param {Node} node
   * @param {Object} [options]
   */
  constructor( node, options ) {

    options = merge( {
      x: 0,
      y: 0,
      width: 1,
      height: 1
    }, options );

    assert && assert( typeof options.x === 'number' && Number.isInteger( options.x ) && isFinite( options.x ) && options.x >= 0 );
    assert && assert( typeof options.y === 'number' && Number.isInteger( options.y ) && isFinite( options.y ) && options.y >= 0 );
    assert && assert( typeof options.width === 'number' && Number.isInteger( options.width ) && isFinite( options.width ) && options.width >= 1 );
    assert && assert( typeof options.height === 'number' && Number.isInteger( options.height ) && isFinite( options.height ) && options.height >= 1 );

    super();

    // @public {OrientationPair.<number>} - These are only set initially, and ignored for the future
    this.position = new OrientationPair( options.x, options.y );
    this.size = new OrientationPair( options.width, options.height );

    // @public {Bounds2} - Set to be the bounds available for the cell
    this.lastAvailableBounds = Bounds2.NOTHING.copy();

    // @public {Bounds2} - Set to be the bounds used by the cell
    this.lastUsedBounds = Bounds2.NOTHING.copy();

    // @private {Node}
    this._node = node;

    this.setOptions( options );

    // @private {function}
    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );

    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  /**
   * @private
   */
  onLayoutOptionsChange() {
    this.setOptions( this.node.layoutOptions || undefined );
  }

  /**
   * @private
   *
   * @param {Object} [options]
   */
  setOptions( options ) {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }

  /**
   * @public
   *
   * @returns {Node}
   */
  get node() {
    return this._node;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMinimumSize( orientation, defaultConfig ) {
    return this.getMinMargin( orientation, defaultConfig ) +
           Math.max(
             this.node[ sizableFlagPair.get( orientation ) ] ? this.node[ minimumSizePair.get( orientation ) ] : this.node[ orientation.size ],
             this.getMinContentSize( orientation, defaultConfig )
           ) +
           this.getMaxMargin( orientation, defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @returns {number}
   */
  getMaximumSize( orientation, defaultConfig ) {
    return this.getMinMargin( orientation, defaultConfig ) +
           Math.min(
             this.getMaxContentSize( orientation, defaultConfig ) || Number.POSITIVE_INFINITY
           ) +
           this.getMaxMargin( orientation, defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   */
  attemptPreferredSize( orientation, defaultConfig, value ) {
    if ( this.node[ sizableFlagPair.get( orientation ) ] ) {
      const minimumSize = this.getMinimumSize( orientation, defaultConfig );
      const maximumSize = this.getMaximumSize( orientation, defaultConfig );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      this.node[ preferredSizePair.get( orientation ) ] = value - this.getMinMargin( orientation, defaultConfig ) - this.getMaxMargin( orientation, defaultConfig );
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable.Align} align
   * @param {GridConfigurable} defaultConfig
   * @param {number} value
   * @param {number} availableSize
   */
  attemptPosition( orientation, align, defaultConfig, value, availableSize ) {
    if ( align === GridConfigurable.Align.ORIGIN ) {
      // TODO: handle layout bounds
      // TODO: OMG this is horribly broken right? We would need to align stuff first
      // TODO: Do a pass to handle origin cells first (and in FLOW too)
      if ( Math.abs( this.node[ orientation.coordinate ] - value ) > CHANGE_POSITION_THRESHOLD ) {
        // TODO: coordinate transform handling, to our ancestorNode!!!!!
        this.node[ orientation.coordinate ] = value;
      }
    }
    else {
      const minMargin = this.getMinMargin( orientation, defaultConfig );
      const maxMargin = this.getMaxMargin( orientation, defaultConfig );
      const extraSpace = availableSize - this.node[ orientation.size ] - minMargin - maxMargin;
      value += minMargin + extraSpace * padRatioMap[ align ];

      if ( Math.abs( this.node[ orientation.minSide ] - value ) > CHANGE_POSITION_THRESHOLD ) {
        // TODO: coordinate transform handling, to our ancestorNode!!!!!
        this.node[ orientation.minSide ] = value;
      }
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   */
  getMinMargin( orientation, defaultConfig ) {
    return this.withDefault( orientation === Orientation.HORIZONTAL ? 'leftMargin' : 'topMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   */
  getMaxMargin( orientation, defaultConfig ) {
    return this.withDefault( orientation === Orientation.HORIZONTAL ? 'rightMargin' : 'bottomMargin', defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   */
  getMinContentSize( orientation, defaultConfig ) {
    return this.withDefault( orientation === Orientation.HORIZONTAL ? 'minContentWidth' : 'minContentHeight', defaultConfig );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable} defaultConfig
   */
  getMaxContentSize( orientation, defaultConfig ) {
    return this.withDefault( orientation === Orientation.HORIZONTAL ? 'maxContentWidth' : 'maxContentHeight', defaultConfig );
  }

  /**
   * @public
   *
   * @param {GridConfigurable} defaultConfig
   * @returns {Bounds2}
   */
  getCellBounds( defaultConfig ) {
    const leftMargin = this.withDefault( 'leftMargin', defaultConfig );
    const rightMargin = this.withDefault( 'rightMargin', defaultConfig );
    const topMargin = this.withDefault( 'topMargin', defaultConfig );
    const bottomMargin = this.withDefault( 'bottomMargin', defaultConfig );

    return this.node.bounds.withOffsets( leftMargin, topMargin, rightMargin, bottomMargin );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {number} index
   * @returns {boolean}
   */
  containsIndex( orientation, index ) {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return index >= position && index < position + size;
  }

  /**
   * @public
   *
   * @param {number} row
   * @returns {boolean}
   */
  containsRow( row ) {
    return this.containsIndex( Orientation.VERTICAL, row );
  }

  /**
   * @public
   *
   * @param {number} column
   * @returns {boolean}
   */
  containsColumn( column ) {
    return this.containsIndex( Orientation.HORIZONTAL, column );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getIndices( orientation ) {
    const position = this.position.get( orientation );
    const size = this.size.get( orientation );
    return _.range( position, position + size );
  }

  /**
   * Releases references
   * @public
   */
  dispose() {
    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'GridCell', GridCell );
export default GridCell;