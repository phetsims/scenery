// Copyright 2002-2013, University of Colorado Boulder

/**
 * VBox arranges the child nodes vertically, and they can be centered, left or right justified.
 * Vertical spacing can be set as a constant or a function which depends on the adjacent nodes.
 *
 * See a dynamic test in scenery\tests\test-vbox.html
 *
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  /**
   *
   * @param options Same as Node.constructor.options with the following additions:
   *
   * spacing: can be a number or a function.  If a number, then it will be the vertical spacing between each object.
   *              If a function, then the function will have the signature function(top,bottom){} which returns the spacing between adjacent pairs of items.
   * align:   How to line up the items horizontally.  One of 'center', 'left' or 'right'.  Defaults to 'center'.
   *
   * @constructor
   */
  scenery.VBox = function VBox( options ) {
    this.boundsListener = this.updateLayout.bind( this );

    // ensure we have a parameter object
    this.options = _.extend( {
      // defaults
      spacing: function() { return 0; },
      align: 'center',

      //By default, update the layout when children are added/removed/resized, see #116
      resize: true
    }, options );

    if ( typeof this.options.spacing === 'number' ) {
      var spacingConstant = this.options.spacing;
      this.options.spacing = function() { return spacingConstant; };
    }

    Node.call( this );

    //See HBox.js
    this.inited = false;
    this.mutate( this.options );
    this.inited = true;
  };
  var VBox = scenery.VBox;

  inherit( Node, VBox, {
    updateLayout: function() {
      if ( !this.updatingLayout ) {
        //Bounds of children are changed in updateLayout, we don't want to stackoverflow so bail if already updating layout
        this.updatingLayout = true;
        var minX = _.min( _.map( this.children, function( child ) {return child.left;} ) );
        var maxX = _.max( _.map( this.children, function( child ) {return child.left + child.width;} ) );
        var centerX = (maxX + minX) / 2;

        //Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
        var y = 0;
        for ( var i = 0; i < this.children.length; i++ ) {
          var child = this.children[i];
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
          y += child.height + this.options.spacing( child, this.children[i + 1] );
        }
        this.updatingLayout = false;
      }
    }
  } );

  //Override the child mutators to updateLayout
  //Have to listen to the child bounds individually because there are a number of possible ways to change the child
  //bounds without changing the overall bounds.
  var overrides = ['insertChild', 'removeChildWithIndex'];
  overrides.forEach( function( override ) {

    //Support up to two args for overrides
    VBox.prototype[override] = function( arg1, arg2 ) {

      //Remove event listeners from any nodes (will be added back later if the node was not removed)
      var vbox = this;
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( child.containsEventListener( 'bounds', vbox.boundsListener ) ) {
            child.removeEventListener( 'bounds', vbox.boundsListener );
          }
        } );
      }

      //Super call
      Node.prototype[override].call( this, arg1, arg2 );

      //Update the layout if it should be dynamic
      if ( this.options.resize || !this.inited ) {
        this.updateLayout();
      }

      //Add event listeners for any current children (if it should be dynamic)
      if ( this.options.resize ) {
        this.children.forEach( function( child ) {
          if ( !child.containsEventListener( 'bounds', vbox.boundsListener ) ) {
            child.addEventListener( 'bounds', vbox.boundsListener );
          }
        } );
      }
    };
  } );

  return VBox;
} );