//  Copyright 2002-2014, University of Colorado Boculder

/**
 * LayoutBox lays out its children in a row, either horizontally or vertically (based on an optional parameter).
 * This approach is preferable to using VBox and HBox (which still exist for backward compatibility)
 * See https://github.com/phetsims/scenery/issues/281
 *
 * @author Sam Reid
 * @author Aaron Davis
 */
define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  /**
   * Main constructor for LayoutBox.
   * @param {Object} [options] Same as Node.constructor.options with the following additions:
   * @constructor
   */
  scenery.LayoutBox = function LayoutBox( options ) {

    Node.call( this );

    this.boundsListener = this.updateLayout.bind( this ); // @private

    // ensure we have a parameter object
    this.options = _.extend( {

      //The default orientation, chosen by popular vote.  At the moment there are around 436 VBox references and 338 HBox references
      orientation: 'vertical',

      // The spacing can be a number or a function.  If a number, then it will be the spacing between each object.
      // If a function, then the function will have the signature function(a,b){} which returns the spacing between adjacent pairs of items.
      spacing: function() { return 0; },

      //How to line up the items
      align: 'center',

      //By default, update the layout when children are added/removed/resized, see #116
      resize: true
    }, options ); // @private

    // Make sure the orientation is legal.  It's better to check this on the this.options instead of the passed in options
    // to keep it closer to the check for the alignment option.
    assert && assert( this.options.orientation === 'vertical' || this.options.orientation === 'horizontal' );

    // Make sure the alignment is allowed, given the orientation.  The check is done here after the final orientation is
    // decided, to reduce logic before the alignment check.
    if ( this.options.orientation === 'vertical' ) {
      assert && assert( this.options.align === 'center' || this.options.align === 'left' || this.options.align === 'right', 'illegal alignment: ' + this.options.align );
    }
    else {
      assert && assert( this.options.align === 'center' || this.options.align === 'top' || this.options.align === 'bottom', 'illegal alignment: ' + this.options.align );
    }

    if ( typeof this.options.spacing === 'number' ) {
      var spacingConstant = this.options.spacing;
      this.options.spacing = function() { return spacingConstant; };
    }

    // Apply the supplied options, including children.
    // The layout calls are triggered if (a) options.resize is set to true or (b) during initialization
    // When true, the this.inited flag signifies that the initial layout is being done.
    this.inited = false; // @private
    this.mutate( this.options );
    this.inited = true;
  };
  var LayoutBox = scenery.LayoutBox;

  return inherit( Node, LayoutBox, {

    // Lay out the child components on startup, or when the children sizes change or when requested by a call to updateLayout
    // @private, do not call directly, use updateLayout
    layout: function() {
      var i = 0;
      var child;

      //Logic for layout out the components.
      //Aaron and Sam looked at factoring this out, but the result looked less readable since each attribute
      //would have to be abstracted over.
      if ( this.options.orientation === 'vertical' ) {
        var minX = _.min( _.map( this.children, function( child ) {return child.left;} ) );
        var maxX = _.max( _.map( this.children, function( child ) {return child.left + child.width;} ) );
        var centerX = (maxX + minX) / 2;

        //Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
        var y = 0;
        for ( i = 0; i < this.children.length; i++ ) {
          child = this.children[ i ];
          child.top = y;

          //Set the position horizontally
          if ( this.options.align === 'left' ) {
            child.left = minX;
          }
          else if ( this.options.align === 'right' ) {
            child.right = maxX;
          }
          else {//default to center
            child.centerX = centerX;
          }

          //Move to the next vertical position.
          y += child.height + this.options.spacing( child, this.children[ i + 1 ] );
        }
      }
      else {
        var minY = _.min( _.map( this.children, function( child ) {return child.top;} ) );
        var maxY = _.max( _.map( this.children, function( child ) {return child.top + child.height;} ) );
        var centerY = (maxY + minY) / 2;

        //Start at x=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {x:number} option.
        var x = 0;
        for ( i = 0; i < this.children.length; i++ ) {
          child = this.children[ i ];
          child.left = x;

          //Set the position horizontally
          if ( this.options.align === 'top' ) {
            child.top = minY;
          }
          else if ( this.options.align === 'bottom' ) {
            child.bottom = maxY;
          }
          else {//default to center
            child.centerY = centerY;
          }

          //Move to the next vertical position.
          x += child.width + this.options.spacing( child, this.children[ i + 1 ] );
        }
      }
    },

    // Update the layout of this VBox. Called automatically during initialization, when children change (if resize is true)
    // or when client wants to call this public method for any reason.
    updateLayout: function() {
      if ( !this.updatingLayout ) {
        //Bounds of children are changed in updateLayout, we don't want to stackoverflow so bail if already updating layout
        this.updatingLayout = true;
        this.layout();
        this.updatingLayout = false;
      }
    },

    //Override the child mutators to updateLayout
    //Have to listen to the child bounds individually because there are a number of possible ways to change the child
    //bounds without changing the overall bounds.
    // @override
    insertChild: function( index, node ) {
      //Support up to two args for overrides

      //Remove event listeners from any nodes (will be added back later if the node was not removed)
      var layoutBox = this;
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.removeEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }

      //Super call
      Node.prototype.insertChild.call( this, index, node );

      // Update the layout (a) if it should be dynamic or (b) during initialization
      if ( this.options.resize || !this.inited ) {
        this.updateLayout();
      }

      //Add event listeners for any current children (if it should be dynamic)
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( !child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.addEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }
    },

    //Overrides the version in Node to listen for bounds changes
    // @override
    removeChildWithIndex: function( node, indexOfChild ) {

      //Remove event listeners from any nodes (will be added back later if the node was not removed)
      var layoutBox = this;
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.removeEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }

      //Super call
      Node.prototype.removeChildWithIndex.call( this, node, indexOfChild );

      // Update the layout (a) if it should be dynamic or (b) during initialization
      if ( this.options.resize || !this.inited ) {
        this.updateLayout();
      }

      //Add event listeners for any current children (if it should be dynamic)
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( !child.containsEventListener( 'bounds', layoutBox.boundsListener ) ) {
            child.addEventListener( 'bounds', layoutBox.boundsListener );
          }
        } );
      }
    }
  } );
} );