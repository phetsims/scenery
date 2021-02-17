// Copyright 2021, University of Colorado Boulder

/**
 * TODO: doc
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import TinyProperty from '../../../axon/js/TinyProperty.js';
import Enumeration from '../../../phet-core/js/Enumeration.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import arrayRemove from '../../../phet-core/js/arrayRemove.js';
import merge from '../../../phet-core/js/merge.js';
import mutate from '../../../phet-core/js/mutate.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import Constraint from './Constraint.js';
import FlowCell from './FlowCell.js';
import FlowConfigurable from './FlowConfigurable.js';

const FLOW_CONSTRAINT_KEYS = [
  'orientation',
  'spacing',
  'justify',
  'wrap'
];

// TODO: Have LayoutBox use this when we're ready
class FlowConstraint extends FlowConfigurable( Constraint ) {
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

    // @private {Array.<FlowCell>}
    this.cells = [];

    // @private {Orientation}
    this._orientation = Orientation.HORIZONTAL;

    // @private {FlowConstraint.Justify}
    this._justify = FlowConstraint.Justify.START;

    // @private {boolean}
    this._wrap = false;

    // @private {number}
    this._spacing = 0;

    // @private {boolean}
    this._excludeInvisibleChildrenFromBounds = true;

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
    mutate( this, FLOW_CONSTRAINT_KEYS, options );

    const updateListener = () => this.updateLayoutAutomatically();

    // Key configuration changes to relayout
    this.changedEmitter.addListener( updateListener );

    // TODO: Add disposal capabilities?
    this.preferredWidthProperty.lazyLink( updateListener );
    this.preferredHeightProperty.lazyLink( updateListener );
  }

  /**
   * @protected
   * @override
   */
  layout() {
    super.layout();

    const orientation = this.orientation;
    const oppositeOrientation = this.orientation.opposite;

    const cells = this.cells.filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisibleChildrenFromBounds || cell.node.visible );
    } );

    if ( !cells.length ) {
      return;
    }

    let lines;
    if ( this.wrap ) {
      lines = [ cells ]; // TODO: wrapping!!!
    }
    else {
      lines = [ cells ];
    }

    // {number}
    const minimumSize = _.max( lines.map( line => {
      return ( line.length - 1 ) * this.spacing + _.sum( line.map( cell => cell.getMinimumSize( orientation, this ) ) );
    } ) );
    const minimumOppositeSize = _.sum( lines.map( line => {
      return _.max( line.map( cell => cell.getMinimumSize( oppositeOrientation, this ) ) );
    } ) );
    this.minimumWidthProperty.value = orientation === Orientation.HORIZONTAL ? minimumSize : minimumOppositeSize;
    this.minimumHeightProperty.value = orientation === Orientation.HORIZONTAL ? minimumOppositeSize : minimumSize;

    // {number}
    const size = Math.max( minimumSize, orientation === Orientation.HORIZONTAL ? this.preferredWidthProperty.value : this.preferredHeightProperty.value );
    // const oppositeSize = Math.max( minimumOppositeSize, orientation === Orientation.HORIZONTAL ? this.preferredHeightProperty.value : this.preferredWidthProperty.value );
    // TODO: opposite-dimension layout

    // TODO: align-content flexbox equivalent

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
      while ( spaceRemaining > 1e7 && ( growableCells = line.filter( ( cell, index ) => {
        if ( cell.grow !== null ) {
          if ( cell.grow === 0 ) {
            return false;
          }
        }
        else if ( this.grow === 0 ) {
          return false;
        }
        return cell._pendingSize < cell.getMaximumSize( orientation, this ) - 1e-7;
      } ) ).length ) {
        const amountEachToGrow = Math.min(
          _.min( growableCells.map( cell => cell.getMaximumSize( orientation, this ) - cell._pendingSize ) ),
          spaceRemaining / growableCells.length
        );

        assert && assert( amountEachToGrow > 1e-11 );

        growableCells.forEach( cell => {
          cell._pendingSize += amountEachToGrow;
        } );
        spaceRemaining -= amountEachToGrow * growableCells.length;
      }

      // Update preferred dimension based on the pending size
      line.forEach( cell => cell.attemptedPreferredSize( orientation, this, cell._pendingSize ) );

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
        cell.positionStart( orientation, this, position );
        position += cell._pendingSize;
      } );
    } );

    // Secondary-direction layout
  }

  /**
   * @public
   *
   * @returns {Orientation}
   */
  get orientation() {
    return this._orientation;
  }

  /**
   * @public
   *
   * @param {Orientation|string} value
   */
  set orientation( value ) {
    if ( value === 'horizontal' ) {
      value = Orientation.HORIZONTAL;
    }
    if ( value === 'vertical' ) {
      value = Orientation.VERTICAL;
    }

    assert && assert( Orientation.includes( value ) );

    if ( this._orientation !== value ) {
      this._orientation = value;

      this.updateLayoutAutomatically();
    }
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
   * @returns {string}
   */
  get excludeInvisibleChildrenFromBounds() {
    return this._excludeInvisibleChildrenFromBounds;
  }

  /**
   * @public
   *
   * @param {Orientation|string} value
   */
  set excludeInvisibleChildrenFromBounds( value ) {
    assert && assert( typeof value === 'boolean' );

    if ( this._excludeInvisibleChildrenFromBounds !== value ) {
      this._excludeInvisibleChildrenFromBounds = value;

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

    this.cells.splice( index, 0, cell );

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

    this.updateLayoutAutomatically();
  }

  /**
   * @public
   *
   * @param {Node} rootNode
   * @param {Object} [options]
   * @returns {FlowConstraint}
   */
  static create( rootNode, options ) {
    return new FlowConstraint( rootNode, options );
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
  'start': FlowConstraint.Justify.START,
  'top': FlowConstraint.Justify.START,
  'left': FlowConstraint.Justify.START,

  'end': FlowConstraint.Justify.END,
  'bottom': FlowConstraint.Justify.END,
  'right': FlowConstraint.Justify.END,

  'center': FlowConstraint.Justify.CENTER,
  'spaceBetween': FlowConstraint.Justify.SPACE_BETWEEN,
  'spaceAround': FlowConstraint.Justify.SPACE_AROUND,
  'spaceEvenly': FlowConstraint.Justify.SPACE_EVENLY
};

scenery.register( 'FlowConstraint', FlowConstraint );
export default FlowConstraint;