// Copyright 2014-2015, University of Colorado Boulder

/**
 * LayoutBox lays out its children in a row, either horizontally or vertically (based on an optional parameter).
 * VBox and HBox are convenience subtypes that specify the orientation.
 * See https://github.com/phetsims/scenery/issues/281
 *
 * @author Sam Reid
 * @author Aaron Davis
 * @author Chris Malley (PixelZoom, Inc.)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  // constants
  var DEFAULT_SPACING = 0;

  /**
   * @param {Object} [options] Same as Node.constructor.options with the following additions:
   * @constructor
   */
  function LayoutBox( options ) {

    if ( options && options.spacing ) {
      assert && assert( typeof options.spacing === 'number', 'LayoutBox requires spacing to be a number, if it is provided' );
    }

    options = _.extend( {

      //The default orientation, chosen by popular vote.  At the moment there are around 436 VBox references and 338 HBox references
      orientation: 'vertical',

      // The spacing between each Node (a number)
      spacing: DEFAULT_SPACING,

      //How to line up the items
      align: 'center',

      //By default, update the layout when children are added/removed/resized, see #116
      resize: true
    }, options ); // @private

    // validate options
    assert && assert( options.orientation === 'vertical' || options.orientation === 'horizontal' );
    if ( options.orientation === 'vertical' ) {
      assert && assert( options.align === 'center' || options.align === 'left' || options.align === 'right', 'illegal alignment: ' + options.align );
    }
    else {
      assert && assert( options.align === 'center' || options.align === 'top' || options.align === 'bottom', 'illegal alignment: ' + options.align );
    }

    this.orientation = options.orientation; // @private
    this.align = options.align; // @private
    this.resize = options.resize; // @private
    this._spacing = options.spacing; // @private {number}

    Node.call( this );

    this.boundsListener = this.updateLayout.bind( this ); // @private
    this.updatingLayout = false; // @private flag used to short-circuit updateLayout and prevent stackoverflow

    // Apply the supplied options, including children.
    // The layout calls are triggered if (a) options.resize is set to true or (b) during initialization
    // When true, the this.inited flag signifies that the initial layout is being done.
    this.inited = false; // @private
    this.mutate( options );
    this.inited = true;
  }

  scenery.register( 'LayoutBox', LayoutBox );

  return inherit( Node, LayoutBox, {

    /**
     * The actual layout logic, typically run from the constructor OR updateLayout().
     * @private
     */
    layout: function() {

      var children = this.getChildren(); // call this once, since it returns a copy
      var i;
      var child;

      // Get the smallest Bounds2 that contains all of our children (triggers bounds validation for all of them)
      var childBounds = this.childBounds;

      // Logic for layout out the components.
      // Aaron and Sam looked at factoring this out, but the result looked less readable since each attribute
      // would have to be abstracted over.
      if ( this.orientation === 'vertical' ) {
        // Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
        var y = 0;
        for ( i = 0; i < children.length; i++ ) {
          child = children[ i ];
          if ( !child.bounds.isValid() ) {
            continue;
          }
          child.top = y;

          // Set the position horizontally
          if ( this.align === 'left' ) {
            child.left = childBounds.minX;
          }
          else if ( this.align === 'right' ) {
            child.right = childBounds.maxX;
          }
          else { // 'center'
            child.centerX = childBounds.centerX;
          }

          // Move to the next vertical position.
          y += child.height + this._spacing;
        }
      }
      else {
        // Start at x=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {x:number} option.
        var x = 0;
        for ( i = 0; i < children.length; i++ ) {
          child = children[ i ];
          if ( !child.bounds.isValid() ) {
            continue;
          }
          child.left = x;

          // Set the position horizontally
          if ( this.align === 'top' ) {
            child.top = childBounds.minY;
          }
          else if ( this.align === 'bottom' ) {
            child.bottom = childBounds.maxY;
          }
          else { // 'center'
            child.centerY = childBounds.centerY;
          }

          // Move to the next horizontal position.
          x += child.width + this._spacing;
        }
      }
    },

    /**
     * Updates the layout of this LayoutBox. Called automatically during initialization, when children change (if
     * resize is true), or when client wants to call this public method for any reason.
     * @public
     */
    updateLayout: function() {
      // Bounds of children are changed in updateLayout, we don't want to stackoverflow, so bail if already updating layout
      if ( !this.updatingLayout ) {
        this.updatingLayout = true;
        this.layout();
        this.updatingLayout = false;
      }
    },

    /**
     * @override - Overrides from Node, so we can listen for bounds changes.
     * We have to listen to the bounds of each child individually, since individual child bounds changes might not
     * trigger an overall bounds change.
     */
    insertChild: function( index, node ) {

      // Super call
      Node.prototype.insertChild.call( this, index, node );

      // Update the layout (a) if it should be dynamic or (b) during initialization
      if ( this.resize || !this.inited ) {
        this.updateLayout();
      }

      if ( this.resize ) {
        node.onStatic( 'bounds', this.boundsListener );
      }
    },

    /**
     * @override - Overrides from Node, so we can listen for bounds changes.
     *
     * @param {Node} node
     * @param {number} indexOfChild
     */
    removeChildWithIndex: function( node, indexOfChild ) {

      if ( this.resize ) {
        node.offStatic( 'bounds', this.boundsListener );
      }

      // Super call
      Node.prototype.removeChildWithIndex.call( this, node, indexOfChild );

      // Update the layout (a) if it should be dynamic or (b) during initialization
      if ( this.resize || !this.inited ) {
        this.updateLayout();
      }
    },

    /**
     * Sets spacing between items in the box.
     * @public
     *
     * @param {number} spacing
     */
    setSpacing: function( spacing ) {

      // Make sure the provided spacing is a number (since we previously allowed number | function here
      assert && assert( typeof spacing === 'number', 'spacing must be a number' );

      if ( this._spacing !== spacing ) {
        this._spacing = spacing;

        // TODO: Do we need to check for if we are resizing?
        this.updateLayout();
      }
    },
    set spacing( value ) {
      this.setSpacing( value );
    },

    /**
     * Gets the spacing between items in the box.
     * @public
     *
     * @returns {number}
     */
    getSpacing: function() {
      return this._spacing;
    },
    get spacing() {
      return this.getSpacing();
    }
  } );
} );