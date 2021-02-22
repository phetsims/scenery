// Copyright 2014-2020, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import GridCell from '../util/GridCell.js';
import GridConstraint from '../util/GridConstraint.js';
import HSizable from '../util/HSizable.js';
import VSizable from '../util/VSizable.js';
import Node from './Node.js';

// GridBox-specific options that can be passed in the constructor or mutate() call.
const GRIDBOX_OPTION_KEYS = [
  'resize' // {boolean} - Whether we should update the layout when children change, see setResize for documentation
].concat( GridConstraint.GRID_CONSTRAINT_OPTION_KEYS ).filter( key => key !== 'excludeInvisible' );

const DEFAULT_OPTIONS = {
  resize: true
};

class GridBox extends HSizable( VSizable( Node ) ) {
  /**
   * @public
   *
   * @param {Object} [options] - GridBox-specific options are documented in GRIDBOX_OPTION_KEYS above, and can be
   *                             provided along-side options for Node.
   */
  constructor( options ) {
    options = merge( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true
    }, options );

    super();

    // @private {Map.<Node,FlowCell>}
    this._cellMap = new Map();

    // @private {GridConstraint}
    this._constraint = new GridConstraint( this, {
      preferredWidthProperty: this.preferredWidthProperty,
      preferredHeightProperty: this.preferredHeightProperty,
      minimumWidthProperty: this.minimumWidthProperty,
      minimumHeightProperty: this.minimumHeightProperty,

      resize: DEFAULT_OPTIONS.resize,
      excludeInvisible: false // Should be handled by the options mutate above
    } );

    this.childInsertedEmitter.addListener( this.onGridBoxChildInserted.bind( this ) );
    this.childRemovedEmitter.addListener( this.onGridBoxChildRemoved.bind( this ) );

    this.mutate( options );
    this._constraint.updateLayout();

    // Adjust the localBounds to be the laid-out area
    this._constraint.layoutBoundsProperty.link( layoutBounds => {
      this.localBounds = layoutBounds;
    } );
  }

  /**
   * @public
   * @override
   *
   * @param {boolean} excludeInvisibleChildrenFromBounds
   */
  setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds ) {
    super.setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds );

    this._constraint.excludeInvisible = excludeInvisibleChildrenFromBounds;
  }

  /**
   * Called when a child is inserted.
   * @private
   *
   * @param {Node} node
   * @param {number} index
   */
  onGridBoxChildInserted( node, index ) {
    const cell = new GridCell( node, node.layoutOptions );
    this._cellMap.set( node, cell );

    this._constraint.addCell( cell );
  }

  /**
   * Called when a child is removed.
   * @private
   *
   * @param {Node} node
   */
  onGridBoxChildRemoved( node ) {

    const cell = this._cellMap.get( node );
    this._cellMap.delete( node );

    this._constraint.removeCell( cell );

    cell.dispose();
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  get resize() {
    return this._constraint.enabled;
  }

  /**
   * @public
   *
   * @returns {number|Array.<number>}
   */
  get spacing() {
    return this._constraint.spacing;
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set spacing( value ) {
    this._constraint.spacing = value;
  }

  /**
   * @public
   *
   * @returns {number|Array.<number>}
   */
  get xSpacing() {
    return this._constraint.xSpacing;
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set xSpacing( value ) {
    this._constraint.xSpacing = value;
  }

  /**
   * @public
   *
   * @returns {number|Array.<number>}
   */
  get ySpacing() {
    return this._constraint.ySpacing;
  }

  /**
   * @public
   *
   * @param {number|Array.<number>} value
   */
  set ySpacing( value ) {
    this._constraint.ySpacing = value;
  }

  /**
   * @public
   *
   * @returns {GridConfigurable.Align|null}
   */
  get xAlign() {
    return this._constraint.xAlign;
  }

  /**
   * @public
   *
   * @param {GridConfigurable.Align|string|null} value
   */
  set xAlign( value ) {
    this._constraint.xAlign = value;
  }

  /**
   * @public
   *
   * @returns {GridConfigurable.Align|null}
   */
  get yAlign() {
    return this._constraint.yAlign;
  }

  /**
   * @public
   *
   * @param {GridConfigurable.Align|string|null} value
   */
  set yAlign( value ) {
    this._constraint.yAlign = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get grow() {
    return this._constraint.grow;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set grow( value ) {
    this._constraint.grow = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get xGrow() {
    return this._constraint.xGrow;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set xGrow( value ) {
    this._constraint.xGrow = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get yGrow() {
    return this._constraint.yGrow;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set yGrow( value ) {
    this._constraint.yGrow = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get margin() {
    return this._constraint.margin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set margin( value ) {
    this._constraint.margin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get xMargin() {
    return this._constraint.xMargin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set xMargin( value ) {
    this._constraint.xMargin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get yMargin() {
    return this._constraint.yMargin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set yMargin( value ) {
    this._constraint.yMargin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get leftMargin() {
    return this._constraint.leftMargin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set leftMargin( value ) {
    this._constraint.leftMargin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get rightMargin() {
    return this._constraint.rightMargin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set rightMargin( value ) {
    this._constraint.rightMargin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get topMargin() {
    return this._constraint.topMargin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set topMargin( value ) {
    this._constraint.topMargin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get bottomMargin() {
    return this._constraint.bottomMargin;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set bottomMargin( value ) {
    this._constraint.bottomMargin = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get minCellWidth() {
    return this._constraint.minCellWidth;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set minCellWidth( value ) {
    this._constraint.minCellWidth = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get minCellHeight() {
    return this._constraint.minCellHeight;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set minCellHeight( value ) {
    this._constraint.minCellHeight = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get maxCellWidth() {
    return this._constraint.maxCellWidth;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set maxCellWidth( value ) {
    this._constraint.maxCellWidth = value;
  }

  /**
   * @public
   *
   * @returns {number|null}
   */
  get maxCellHeight() {
    return this._constraint.maxCellHeight;
  }

  /**
   * @public
   *
   * @param {number|null} value
   */
  set maxCellHeight( value ) {
    this._constraint.maxCellHeight = value;
  }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @protected
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
GridBox.prototype._mutatorKeys = HSizable( Node ).prototype._mutatorKeys.concat( VSizable( Node ).prototype._mutatorKeys ).concat( GRIDBOX_OPTION_KEYS );

// @public {Object}
GridBox.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'GridBox', GridBox );
export default GridBox;