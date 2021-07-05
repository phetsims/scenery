// Copyright 2014-2021, University of Colorado Boulder

/**
 * LayoutBox lays out its children in a row, either horizontally or vertically (based on an optional parameter).
 * VBox and HBox are convenience subtypes that specify the orientation.
 * See https://github.com/phetsims/scenery/issues/281
 *
 * @author Sam Reid
 * @author Aaron Davis
 * @author Chris Malley (PixelZoom, Inc.)
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import SceneryConstants from '../SceneryConstants.js';
import Node from './Node.js';

// constants
const DEFAULT_SPACING = 0;

// LayoutBox-specific options that can be passed in the constructor or mutate() call.
const LAYOUT_BOX_OPTION_KEYS = [
  'orientation', // {string} - 'horizontal' or 'vertical', see setOrientation for documentation
  'spacing', // {number} - Spacing between each Node, see setSpacing for documentation
  'align', // {string} - How to line up items, see setAlign for documentation
  'resize' // {boolean} - Whether we should update the layout when children change, see setResize for documentation
];

// The position (left/top) property name on the primary axis
const LAYOUT_POSITION = {
  vertical: 'top',
  horizontal: 'left'
};

// The size (width/height) property name on the primary axis
const LAYOUT_DIMENSION = {
  vertical: 'height',
  horizontal: 'width'
};

// The alignment property name on the secondary axis
const LAYOUT_ALIGNMENT = {
  vertical: {
    left: 'left',
    center: 'centerX',
    right: 'right',
    origin: 'x'
  },
  horizontal: {
    top: 'top',
    center: 'centerY',
    bottom: 'bottom',
    origin: 'y'
  }
};

class LayoutBox extends Node {
  /**
   * @public
   *
   * @param {Object} [options] - LayoutBox-specific options are documented in LAYOUT_BOX_OPTION_KEYS above, and can be
   *                             provided along-side options for Node.
   */
  constructor( options ) {
    options = merge( {

      // Allow dynamic layout by default, see https://github.com/phetsims/joist/issues/608
      excludeInvisibleChildrenFromBounds: true,

      disabledOpacity: SceneryConstants.DISABLED_OPACITY
    }, options );

    super();

    // @private {string} - Either 'vertical' or 'horizontal'. The default chosen by popular vote (with more references).
    //                     see setOrientation() for more documentation.
    this._orientation = 'vertical';

    // @private {number} - Spacing between nodes, see setSpacing() for more documentation.
    this._spacing = DEFAULT_SPACING;

    // @private {string} - Either 'left', 'center', 'right' or 'origin' for vertical layout, or 'top', 'center',
    //                     'bottom' or 'origin' for horizontal layout, which controls positioning of nodes on the other
    //                     axis. See setAlign() for more documentation.
    this._align = 'center';

    // @private {boolean} - Whether we'll layout after children are added/removed/resized, see #116. See setResize()
    //                      for more documentation.
    this._resize = true;

    // @private {function} - If resize:true, will be called whenever a child has its bounds change
    this._updateLayoutListener = this.updateLayoutAutomatically.bind( this );

    // @private {boolean} - Prevents layout() from running while true. Generally will be unlocked and laid out.
    this._updateLayoutLocked = false;

    this.childInsertedEmitter.addListener( this.onLayoutBoxChildInserted.bind( this ) );
    this.childRemovedEmitter.addListener( this.onLayoutBoxChildRemoved.bind( this ) );
    this.childrenChangedEmitter.addListener( this.onLayoutBoxChildrenChanged.bind( this ) );

    // @private {boolean} - We'll ignore the resize flag while running the initial mutate.
    this._layoutMutating = true;

    this.mutate( options );

    this._layoutMutating = false;
  }


  /**
   * Given the current children, determines what bounds should they be aligned inside of.
   * @private
   *
   * Triggers bounds validation for all children
   *
   * @returns {Bounds2}
   */
  getAlignmentBounds() {
    // Construct a Bounds2 at the origin, but with the maximum width/height of the children.
    let maxWidth = Number.NEGATIVE_INFINITY;
    let maxHeight = Number.NEGATIVE_INFINITY;

    for ( let i = 0; i < this._children.length; i++ ) {
      const child = this._children[ i ];

      if ( this.isChildIncludedInLayout( child ) ) {
        maxWidth = Math.max( maxWidth, child.width );
        maxHeight = Math.max( maxHeight, child.height );
      }
    }
    return new Bounds2( 0, 0, maxWidth, maxHeight );
  }

  /**
   * The actual layout logic, typically run from the constructor OR updateLayout().
   * @private
   */
  layout() {
    const children = this._children;

    // The position (left/top) property name on the primary axis
    const layoutPosition = LAYOUT_POSITION[ this._orientation ];

    // The size (width/height) property name on the primary axis
    const layoutDimension = LAYOUT_DIMENSION[ this._orientation ];

    // The alignment (left/right/bottom/top/centerX/centerY) property name on the secondary axis
    const layoutAlignment = LAYOUT_ALIGNMENT[ this._orientation ][ this._align ];

    // The bounds that children will be aligned in (on the secondary axis)
    const alignmentBounds = this.getAlignmentBounds();

    // Starting at 0, position the children
    let position = 0;
    for ( let i = 0; i < children.length; i++ ) {
      const child = children[ i ];

      if ( this.isChildIncludedInLayout( child ) ) {
        child[ layoutPosition ] = position;
        child[ layoutAlignment ] = this._align === 'origin' ? 0 : alignmentBounds[ layoutAlignment ];
        position += child[ layoutDimension ] + this._spacing; // Move forward by the node's size, including spacing
      }
    }
  }

  /**
   * Updates the layout of this LayoutBox. Called automatically during initialization, when children change (if
   * resize is true), or when client wants to call this public method for any reason.
   * @public
   */
  updateLayout() {
    // Since we trigger bounds changes in our children during layout, we don't want to trigger layout off of those
    // changes, causing a stack overflow.
    if ( !this._updateLayoutLocked ) {
      this._updateLayoutLocked = true;
      this.layout();
      this._updateLayoutLocked = false;
    }
  }

  /**
   * Called when we attempt to automatically layout components.
   * @private
   */
  updateLayoutAutomatically() {
    if ( this._layoutMutating || this._resize ) {
      this.updateLayout();
    }
  }

  /**
   * Called when a child is inserted.
   * @private
   *
   * @param {Node} node
   */
  onLayoutBoxChildInserted( node ) {
    if ( this._resize ) {
      node.boundsProperty.lazyLink( this._updateLayoutListener );
      node.visibleProperty.lazyLink( this._updateLayoutListener );
    }
  }

  /**
   * Called when a child is removed.
   * @private
   *
   * @param {Node} node
   */
  onLayoutBoxChildRemoved( node ) {
    if ( this._resize ) {
      node.boundsProperty.unlink( this._updateLayoutListener );
      node.visibleProperty.unlink( this._updateLayoutListener );
    }
  }

  /**
   * Called on change of children (child added, removed, order changed, etc.)
   * @private
   */
  onLayoutBoxChildrenChanged() {
    if ( this._resize ) {
      this.updateLayoutAutomatically();
    }
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
   * @returns {LayoutBox} - Returns 'this' reference, for chaining
   */
  setChildren( children ) {
    // If the layout is already locked, we need to bail and only call Node's setChildren.
    if ( this._updateLayoutLocked ) {
      return super.setChildren( children );
    }

    const oldChildren = this.getChildren(); // defensive copy

    // Lock layout while the children are removed and added
    this._updateLayoutLocked = true;
    super.setChildren( children );
    this._updateLayoutLocked = false;

    // Determine if the children array has changed. We'll gain a performance benefit by not triggering layout when
    // the children haven't changed.
    if ( !_.isEqual( oldChildren, children ) ) {
      this.updateLayoutAutomatically();
    }

    return this;
  }

  /**
   * If this is set to true, child nodes that are invisible will NOT contribute to the bounds of this node.
   * @public
   * @override
   *
   * The default is for child nodes bounds' to be included in this node's bounds, but that would in general be a
   * problem for layout containers or other situations, see https://github.com/phetsims/joist/issues/608.
   *
   * @param {boolean} excludeInvisibleChildrenFromBounds
   */
  setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds ) {
    super.setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds );

    // If we have invisible children, we'll likely need to update our layout,
    // see https://github.com/phetsims/sun/issues/594
    if ( _.some( this._children, child => !child.visible ) ) {
      this.updateLayoutAutomatically();
    }
  }

  /**
   * Sets the orientation of the LayoutBox (the axis along which nodes will be placed, separated by spacing).
   * @public
   *
   * @param {string} orientation - Should be either 'vertical' or 'horizontal'
   * @returns {Node} - For chaining
   */
  setOrientation( orientation ) {
    assert && assert( this._orientation === 'vertical' || this._orientation === 'horizontal' );

    if ( this._orientation !== orientation ) {
      this._orientation = orientation;

      this.updateLayout();
    }

    return this;
  }

  set orientation( value ) { this.setOrientation( value ); }

  /**
   * Returns the current orientation.
   * @public
   *
   * See setOrientation for more documentation on the orientation.
   *
   * @returns {string}
   */
  getOrientation() {
    return this._orientation;
  }

  get orientation() { return this.getOrientation(); }

  /**
   * Sets spacing between items in the LayoutBox.
   * @public
   *
   * @param {number} spacing
   * @returns {Node} - For chaining
   */
  setSpacing( spacing ) {
    assert && assert( typeof spacing === 'number' && isFinite( spacing ),
      'spacing must be a finite number' );

    if ( this._spacing !== spacing ) {
      this._spacing = spacing;

      this.updateLayout();
    }

    return this;
  }

  set spacing( value ) { this.setSpacing( value ); }

  /**
   * Gets the spacing between items in the LayoutBox.
   * @public
   *
   * See setSpacing() for more documentation on spacing.
   *
   * @returns {number}
   */
  getSpacing() {
    return this._spacing;
  }

  get spacing() { return this.getSpacing(); }

  /**
   * Sets the alignment of the LayoutBox.
   * @public
   *
   * Determines how children of this LayoutBox will be positioned along the opposite axis from the orientation.
   *
   * For vertical alignments (the default), the following align values are allowed:
   * - left
   * - center
   * - right
   * - origin - The x value of each child will be set to 0.
   *
   * For horizontal alignments, the following align values are allowed:
   * - top
   * - center
   * - bottom
   * - origin - The y value of each child will be set to 0.
   *
   * @param {string} align
   * @returns {Node} - For chaining
   */
  setAlign( align ) {
    if ( assert ) {
      if ( this._orientation === 'vertical' ) {
        assert( this._align === 'left' || this._align === 'center' || this._align === 'right' || this._align === 'origin',
          `Illegal vertical LayoutBox alignment: ${align}` );
      }
      else {
        assert( this._align === 'top' || this._align === 'center' || this._align === 'bottom' || this._align === 'origin',
          `Illegal horizontal LayoutBox alignment: ${align}` );
      }
    }

    if ( this._align !== align ) {
      this._align = align;

      this.updateLayout();
    }

    return this;
  }

  set align( value ) { this.setAlign( value ); }

  /**
   * Returns the current alignment.
   * @public
   *
   * See setAlign for more documentation on the orientation.
   *
   * @returns {string}
   */
  getAlign() {
    return this._align;
  }

  get align() { return this.getAlign(); }

  /**
   * Sets whether this LayoutBox will trigger layout when children are added/removed/resized.
   * @public
   *
   * Layout will always still be triggered on orientation/align/spacing changes.
   *
   * @param {boolean} resize
   * @returns {Node} - For chaining
   */
  setResize( resize ) {
    assert && assert( typeof resize === 'boolean', 'resize should be a boolean' );

    if ( this._resize !== resize ) {
      this._resize = resize;

      // Add or remove listeners, based on how resize switched
      for ( let i = 0; i < this._children.length; i++ ) {
        const child = this._children[ i ];

        // If we are now resizable, we need to add listeners to every child
        if ( resize ) {
          child.boundsProperty.lazyLink( this._updateLayoutListener );
          child.visibleProperty.lazyLink( this._updateLayoutListener );
        }
        // Otherwise we are now not resizeable, and need to remove the listeners
        else {
          child.boundsProperty.unlink( this._updateLayoutListener );
          child.visibleProperty.unlink( this._updateLayoutListener );
        }
      }

      // Only trigger an update if we switched TO resizing
      this.updateLayoutAutomatically();
    }

    return this;
  }

  set resize( value ) { this.setResize( value ); }

  /**
   * Returns whether this LayoutBox will trigger layout when children are added/removed/resized.
   * @public
   *
   * See setResize() for more documentation on spacing.
   *
   * @returns {boolean}
   */
  isResize() {
    return this._resize;
  }

  get resize() { return this.isResize(); }
}

/**
 * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
 * order they will be evaluated in.
 * @protected
 *
 * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
 *       cases that may apply.
 */
LayoutBox.prototype._mutatorKeys = LAYOUT_BOX_OPTION_KEYS.concat( Node.prototype._mutatorKeys );

scenery.register( 'LayoutBox', LayoutBox );

export default LayoutBox;