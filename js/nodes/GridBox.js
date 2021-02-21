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
].concat( GridConstraint.GRID_CONSTRAINT_OPTION_KEYS );

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
   * @param {boolean} value
   */
  set resize( value ) {
    this._constraint.enabled = value;
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
GridBox.prototype._mutatorKeys = HSizable( Node ).prototype._mutatorKeys.concat( VSizable( Node ).prototype._mutatorKeys ).concat( GRIDBOX_OPTION_KEYS ).filter( key => key !== 'excludeInvisible' );

// @public {Object}
GridBox.DEFAULT_OPTIONS = DEFAULT_OPTIONS;

scenery.register( 'GridBox', GridBox );
export default GridBox;