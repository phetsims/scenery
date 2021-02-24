// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import merge from '../../../phet-core/js/merge.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import FlowCell from './FlowCell.js';
import FlowConstraint from './FlowConstraint.js';
import HSizable from './HSizable.js';
import VSizable from './VSizable.js';

// FlowBox-specific options that can be passed in the constructor or mutate() call.
const FLOWBOX_OPTION_KEYS = [
  'resize' // {boolean} - Whether we should update the layout when children change, see setResize for documentation
].concat( FlowConstraint.FLOW_CONSTRAINT_OPTION_KEYS ).filter( key => key !== 'excludeInvisible' );

const DEFAULT_OPTIONS = {
  orientation: 'horizontal',
  spacing: 0,
  align: 'center',
  resize: true
};

class FlowBox extends HSizable( VSizable( Node ) ) {
  /**
   * @param {Object} [options] - FlowBox-specific options are documented in FLOWBOX_OPTION_KEYS above, and can be
   *                             provided along-side options for Node.
   */
  constructor( options ) {
    options = merge( {
      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true
    }, options );

    super();

    // @private {FlowConstraint}
    this._constraint = new FlowConstraint( this, {
      preferredWidthProperty: this.preferredWidthProperty,
      preferredHeightProperty: this.preferredHeightProperty,
      minimumWidthProperty: this.minimumWidthProperty,
      minimumHeightProperty: this.minimumHeightProperty,

      orientation: DEFAULT_OPTIONS.orientation,
      spacing: DEFAULT_OPTIONS.spacing,
      align: DEFAULT_OPTIONS.align,
      resize: DEFAULT_OPTIONS.resize,
      excludeInvisible: false // Should be handled by the options mutate above
    } );

    // @private {Map.<Node,FlowCell>}
    this._cellMap = new Map();

    this.childInsertedEmitter.addListener( this.onFlowBoxChildInserted.bind( this ) );
    this.childRemovedEmitter.addListener( this.onFlowBoxChildRemoved.bind( this ) );
    this.childrenReorderedEmitter.addListener( this.onFlowBoxChildrenReordered.bind( this ) );
    this.childrenChangedEmitter.addListener( this.onFlowBoxChildrenChanged.bind( this ) );

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
  onFlowBoxChildInserted( node, index ) {
    const cell = new FlowCell( node, node.layoutOptions );
    this._cellMap.set( node, cell );

    this._constraint.insertCell( index, cell );
  }

  /**
   * Called when a child is removed.
   * @private
   *
   * @param {Node} node
   */
  onFlowBoxChildRemoved( node ) {

    const cell = this._cellMap.get( node );
    this._cellMap.delete( node );

    this._constraint.removeCell( cell );

    cell.dispose();
  }

  /**
   * Called when children are rearranged
   * @private
   *
   * @param {number} minChangeIndex
   * @param {number} maxChangeIndex
   */
  onFlowBoxChildrenReordered( minChangeIndex, maxChangeIndex ) {
    this._constraint.reorderCells(
      this._children.slice( minChangeIndex, maxChangeIndex + 1 ).map( node => this._cellMap.get( node ) ),
      minChangeIndex,
      maxChangeIndex
    );
  }

  /**
   * Called on change of children (child added, removed, order changed, etc.)
   * @private
   */
  onFlowBoxChildrenChanged() {
    this._constraint.updateLayoutAutomatically();
  }

  /**
   * Sets the children of the Node to be equivalent to the passed-in array of Nodes. Does this by removing all current
   * children, and adding in children from the array.
   * @public
   * @override
   *
   * Overridden so we can group together setChildren() and only update layout (a) at the end, and (b) if there
   * are changes.
   *
   * @param {Array.<Node>} children
   * @returns {FlowBox} - Returns 'this' reference, for chaining
   */
  setChildren( children ) {
    // If the layout is already locked, we need to bail and only call Node's setChildren.
    if ( this._constraint.isLocked ) {
      return super.setChildren( children );
    }

    const oldChildren = this.getChildren(); // defensive copy

    // Lock layout while the children are removed and added
    this._constraint.lock();
    super.setChildren( children );
    this._constraint.unlock();

    // Determine if the children array has changed. We'll gain a performance benefit by not triggering layout when
    // the children haven't changed.
    if ( !_.isEqual( oldChildren, children ) ) {
      this._constraint.updateLayoutAutomatically();
    }

    return this;
  }

  /**
   * @public
   *
   * @param {Node} node
   * @returns {FlowCell}
   */
  getCell( node ) {
    return this._cellMap.get( node );
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
   * @param {boolean} value
   */
  set resize( value ) {
    this._constraint.enabled = value;
  }

  /**
   * @public
   *
   * @returns {Orientation}
   */
  get orientation() {
    return this._constraint.orientation;
  }

  /**
   * @public
   *
   * @param {Orientation|string} value
   */
  set orientation( value ) {
    this._constraint.orientation = value;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get spacing() {
    return this._constraint.spacing;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set spacing( value ) {
    this._constraint.spacing = value;
  }

  /**
   * @public
   *
   * @returns {number}
   */
  get lineSpacing() {
    return this._constraint.lineSpacing;
  }

  /**
   * @public
   *
   * @param {number} value
   */
  set lineSpacing( value ) {
    this._constraint.lineSpacing = value;
  }

  /**
   * @public
   *
   * @returns {FlowConstraint.Justify}
   */
  get justify() {
    return this._constraint.justify;
  }

  /**
   * @public
   *
   * @param {FlowConstraint.Justify|string} value
   */
  set justify( value ) {
    this._constraint.justify = value;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  get wrap() {
    return this._constraint.wrap;
  }

  /**
   * @public
   *
   * @param {boolean} value
   */
  set wrap( value ) {
    this._constraint.wrap = value;
  }

  /**
   * @public
   *
   * @returns {FlowConfigurable.Align|null}
   */
  get align() {
    return this._constraint.align;
  }

  /**
   * @public
   *
   * @param {FlowConfigurable.Align|string|null} value
   */
  set align( value ) {
    this._constraint.align = value;
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

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this._constraint.dispose();

    this._cellMap.values().forEach( cell => {
      cell.dispose();
    } );

    super.dispose();
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
FlowBox.prototype._mutatorKeys = HSizable( Node ).prototype._mutatorKeys.concat( VSizable( Node ).prototype._mutatorKeys ).concat( FLOWBOX_OPTION_KEYS );

// @public {Object}
FlowBox.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'FlowBox', FlowBox );
export default FlowBox;