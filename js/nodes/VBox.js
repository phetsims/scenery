// Copyright 2002-2012, University of Colorado

/**
 * VBox arranges the child nodes vertically, and they can be centered, left or right justified.
 * Vertical spacing can be set as a constant or a function which depends on the adjacent nodes.
 * TODO: add an option (not enabled by default) to update layout when children or children bounds change
 *
 * @author Sam Reid
 */

define( function( require ) {
  "use strict";
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  var objectCreate = require( 'SCENERY/util/Util' ).objectCreate; // i.e. Object.create

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
    // ensure we have a parameter object
    this.options = options = _.extend( {
      // defaults
      spacing: function() { return 0; },
      align: 'center'
    }, options );
    
    if ( typeof options.spacing === 'number' ) {
      var spacingConstant = options.spacing;
      options.spacing = function() { return spacingConstant; };
    }

    Node.call( this, options );
    this.updateLayout();
  };
  var VBox = scenery.VBox;

  VBox.prototype = objectCreate( Node.prototype );

  VBox.prototype.updateLayout = function() {
    var minX = _.min( _.map( this.children, function( child ) {return child.x;} ) );
    var maxX = _.max( _.map( this.children, function( child ) {return child.x + child.width;} ) );
    var centerX = (maxX + minX) / 2;

    //Start at y=0 in the coordinate frame of this node.  Not possible to set this through the spacing option, instead just set it with the {y:number} option.
    var y = 0;
    for ( var i = 0; i < this.children.length; i++ ) {
      var child = this.children[i];
      child.y = y;

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
  };
  VBox.prototype.constructor = VBox;

  return VBox;
} );
