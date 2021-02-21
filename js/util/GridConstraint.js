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
  'excludeInvisible'
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

    // const preferredWidth = this.preferredWidthProperty.value;
    // const preferredHeight = this.preferredHeightProperty.value;

    const cells = this.cells.values().filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
    } );

    if ( !cells.length ) {
      this.layoutBoundsProperty.value = Bounds2.NOTHING;
      return;
    }



    // TODO: you know, some layout and stuff

    // We're taking up these layout bounds (nodes could use them for localBounds)
    this.layoutBoundsProperty.value = new Bounds2( 0, 0, 0, 0 ); // TODO: layoutBounds
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
   * @param {GridCell} cell
   */
  addCell( cell ) {
    assert && assert( cell instanceof GridCell );
    assert && assert( !_.includes( this.cells, cell ) );

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

    this.cells.values().forEach( cell => this.removeCell( cell ) );

    super.dispose();
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