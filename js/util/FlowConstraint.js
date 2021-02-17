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
import Constraint from './Constraint.js';
import FlowCell from './FlowCell.js';
import FlowConfigurable from './FlowConfigurable.js';

const FLOW_CONSTRAINT_OPTION_KEYS = [
  'orientation',
  'spacing',
  'lineSpacing',
  'justify',
  'wrap',
  'excludeInvisible'
].concat( FlowConfigurable.FLOW_CONFIGURABLE_OPTION_KEYS );

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
    this.layoutBoundsProperty = new Property( Bounds2.NOTHING );

    this.preferredWidthProperty = options.preferredWidthProperty;
    this.preferredHeightProperty = options.preferredHeightProperty;
    this.minimumWidthProperty = options.minimumWidthProperty;
    this.minimumHeightProperty = options.minimumHeightProperty;

    this.setConfigToBaseDefault();
    this.mutateConfigurable( options );
    mutate( this, FLOW_CONSTRAINT_OPTION_KEYS, options );

    // Key configuration changes to relayout
    this.changedEmitter.addListener( this._updateLayoutListener );

    // TODO: Add disposal capabilities?
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

    const cells = this.cells.filter( cell => {
      // TODO: Also don't lay out disconnected nodes!!!!
      return cell.node.bounds.isValid() && ( !this._excludeInvisible || cell.node.visible );
    } );

    if ( !cells.length ) {
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
    const oppositeSize = Math.max( minimumCurrentOppositeSize, preferredOppositeSize || 0 );

    // We're taking up these layout bounds (nodes could use them for localBounds)
    // TODO: How to handle this with align:origin!!!
    this.layoutBoundsProperty.value = new Bounds2( 0, 0, orientation === Orientation.HORIZONTAL ? size : oppositeSize, orientation === Orientation.HORIZONTAL ? oppositeSize : size );

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
      while ( spaceRemaining > 1e-7 && ( growableCells = line.filter( cell => {
        const grow = cell.withDefault( 'grow', this );
        if ( grow === 0 ) {
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
          cell.attemptedPreferredSize( oppositeOrientation, this, maximumSize );
          cell.positionStart( oppositeOrientation, this, secondaryPosition );
        }
        else {
          cell.attemptedPreferredSize( oppositeOrientation, this, size );

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
      } );

      secondaryPosition += maximumSize + this.lineSpacing;
    } );
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

    // TODO: handle disposing here?
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

// @public {Array.<string>}
FlowConstraint.FLOW_CONSTRAINT_OPTION_KEYS = FLOW_CONSTRAINT_OPTION_KEYS;

scenery.register( 'FlowConstraint', FlowConstraint );
export default FlowConstraint;