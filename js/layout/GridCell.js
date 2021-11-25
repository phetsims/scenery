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
import { scenery, GridConfigurable } from '../imports.js';

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
   * @param {GridConstraint} constraint
   * @param {Node} node
   * @param {Object} [options]
   */
  constructor( constraint, node, options ) {

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

    // @private {GridConstraint}
    this._constraint = constraint;

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
   * @public
   *
   * @returns {GridConfigurable.Align}
   */
  get effectiveXAlign() {
    return this._xAlign !== null ? this._xAlign : this._constraint._xAlign;
  }

  /**
   * @public
   *
   * @returns {GridConfigurable.Align}
   */
  get effectiveYAlign() {
    return this._yAlign !== null ? this._yAlign : this._constraint._yAlign;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getEffectiveAlign( orientation ) {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXAlign : this.effectiveYAlign;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveLeftMargin() {
    return this._leftMargin !== null ? this._leftMargin : this._constraint._leftMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveRightMargin() {
    return this._rightMargin !== null ? this._rightMargin : this._constraint._rightMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveTopMargin() {
    return this._topMargin !== null ? this._topMargin : this._constraint._topMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveBottomMargin() {
    return this._bottomMargin !== null ? this._bottomMargin : this._constraint._bottomMargin;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getEffectiveMinMargin( orientation ) {
    return orientation === Orientation.HORIZONTAL ? this.effectiveLeftMargin : this.effectiveTopMargin;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getEffectiveMaxMargin( orientation ) {
    return orientation === Orientation.HORIZONTAL ? this.effectiveRightMargin : this.effectiveBottomMargin;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveXGrow() {
    return this._xGrow !== null ? this._xGrow : this._constraint._xGrow;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveYGrow() {
    return this._yGrow !== null ? this._yGrow : this._constraint._yGrow;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getEffectiveGrow( orientation ) {
    return orientation === Orientation.HORIZONTAL ? this.effectiveXGrow : this.effectiveYGrow;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMinContentWidth() {
    return this._minContentWidth !== null ? this._minContentWidth : this._constraint._minContentWidth;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMinContentHeight() {
    return this._minContentHeight !== null ? this._minContentHeight : this._constraint._minContentHeight;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getEffectiveMinContent( orientation ) {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMinContentWidth : this.effectiveMinContentHeight;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMaxContentWidth() {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._constraint._maxContentWidth;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get effectiveMaxContentHeight() {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._constraint._maxContentHeight;
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getEffectiveMaxContent( orientation ) {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMaxContentWidth : this.effectiveMaxContentHeight;
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
   * @returns {number}
   */
  getMinimumSize( orientation ) {
    return this.getEffectiveMinMargin( orientation ) +
           Math.max(
             this.node[ sizableFlagPair.get( orientation ) ] ? this.node[ minimumSizePair.get( orientation ) ] : this.node[ orientation.size ],
             this.getEffectiveMinContent( orientation )
           ) +
           this.getEffectiveMaxMargin( orientation );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @returns {number}
   */
  getMaximumSize( orientation ) {
    return this.getEffectiveMinMargin( orientation ) +
           Math.min(
             this.getEffectiveMaxContent( orientation ) || Number.POSITIVE_INFINITY
           ) +
           this.getEffectiveMaxMargin( orientation );
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {number} value
   */
  attemptPreferredSize( orientation, value ) {
    if ( this.node[ sizableFlagPair.get( orientation ) ] ) {
      const minimumSize = this.getMinimumSize( orientation );
      const maximumSize = this.getMaximumSize( orientation );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      this.node[ preferredSizePair.get( orientation ) ] = value - this.getEffectiveMinMargin( orientation ) - this.getEffectiveMaxMargin( orientation );
    }
  }

  /**
   * @public
   *
   * @param {Orientation} orientation
   * @param {GridConfigurable.Align} align
   * @param {number} value
   * @param {number} availableSize
   */
  attemptPosition( orientation, align, value, availableSize ) {
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
      const minMargin = this.getEffectiveMinMargin( orientation );
      const maxMargin = this.getEffectiveMaxMargin( orientation );
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
   * @returns {Bounds2}
   */
  getCellBounds() {
    return this.node.bounds.withOffsets(
      this.effectiveLeftMargin,
      this.effectiveTopMargin,
      this.effectiveRightMargin,
      this.effectiveBottomMargin );
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